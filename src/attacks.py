import gc
import os
from os import sys
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.privacy_tests import utils
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import \
    advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import \
    membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
    AttackInputData, SlicingSpec)

from model import Model
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder, get_ds_as_numpy
from util import pickle_object, unpickle_object


class AmiaAttack:
    """Implementation for multi class advanced mia attack.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used

    Code mostly copied from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py
    """

    def __init__(
        self,
        model: Model,
        ds: AbstractDataset,
        result_path: str,
        shadow_model_dir: str,
        include_mia: bool = False,
        num_shadow_models: int = 16,
    ):
        """Initialize MiaAttack class.

        Paramter:
        ---------
        model : CNNModel
        dataset : AbstractDataset - When instatiating a Dataset, a validation dataset is not needed, instead increase the train sice when specifying the train_val_test_split
        """
        self.models_dir = shadow_model_dir
        check_create_folder(self.models_dir)

        self.result_path = result_path
        check_create_folder(self.result_path)

        self.cnn_model: Model = model

        if ds.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            sys.exit(1)

        if ds.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            sys.exit(1)

        self.include_mia = include_mia

        self.ds = ds

        self.num_shadow_models = num_shadow_models

        self.in_indices: Optional[
            list
        ] = None  # a list of in-training indices for all models
        self.stat: Optional[list] = None  # a list of statistics for all models
        self.losses: Optional[list] = None  # a list of losses for all models
        self.attack_result_list: Optional[list] = None
        self.attack_baseline_result_list: Optional[list] = None

        self.num_training_samples: int = 0

        # try loading the in_indices if it was saved
        self.numpy_path = os.path.join(self.models_dir, "data")
        check_create_folder(self.numpy_path)

        self.in_indices_filename = os.path.join(self.numpy_path, "in_indices.pckl")
        self.stat_filename = os.path.join(self.numpy_path, "model_stat.pckl")
        self.loss_filename = os.path.join(self.numpy_path, "model_loss.pckl")

        self.attack_statistics_folder: str = os.path.join(
            self.result_path, "attack-statistics"
        )
        check_create_folder(self.attack_statistics_folder)

        self.attack_result_list_filename = os.path.join(
            self.attack_statistics_folder,
            "pickles",
            f"{ds.dataset_name}_attack_results.pckl",
        )
        self.attack_baseline_result_list_filename = os.path.join(
            self.attack_statistics_folder,
            "pickles",
            f"{ds.dataset_name}_attack_baseline_results.pckl",
        )

    def train_load_shadow_models(
        self,
        force_retraining: bool = False,
        force_recalculation: bool = False,
        num_microbatch: Optional[int] = None,
    ):
        """Trains, or if shadow models are already trained and saved, loads shadow models from filesystem.

        After training/ loading the shadow models statistics and losses are calulcated over all shadow models.

        """
        (train_samples, train_labels) = self.ds.get_train_ds_as_numpy()
        (test_samples, test_labels) = self.ds.get_test_ds_as_numpy()

        test_labels = tf.one_hot(test_labels, depth=self.ds.num_classes)

        # convert to one-hot encoding
        train_labels = tf.one_hot(train_labels, depth=self.ds.num_classes).numpy()

        self.num_training_samples = len(train_samples)

        self.stat = unpickle_object(self.stat_filename)
        self.losses = unpickle_object(self.loss_filename)
        self.in_indices = unpickle_object(self.in_indices_filename)

        loaded_indices: bool = False

        if self.in_indices is not None:
            loaded_indices = True

        if force_retraining:
            self.in_indices = []
            loaded_indices = False

        if force_recalculation or force_retraining:
            self.stat = []
            self.losses = []

        if (
            self.in_indices is not None
            and self.stat is not None
            and self.losses is not None
        ):
            if len(self.in_indices) > 0 and len(self.stat) > 0 and len(self.losses) > 0:
                print(
                    "Loaded in_indices file, stat file and loss file, do not need to load models."
                )
                return
        else:
            self.stat = []
            self.losses = []
            if self.in_indices is None:
                self.in_indices = []

        for i in range(self.num_shadow_models + 1):
            print(f"Creating shadow model {i} and its statistics")

            model_path = os.path.join(
                self.models_dir,
                f"shadow_model_{i}_lr{self.cnn_model.learning_rate}_b{self.cnn_model.batch_size}_e{self.cnn_model.epochs}",
            )

            if loaded_indices:
                keep = self.in_indices[i]
            else:
                keep = self.ds.generate_class_dependent_in_indices(
                    train_samples, train_labels, reduction_factor=0.5
                )
                self.in_indices.append(keep)

            # prepare model for new train iteration
            self.cnn_model.model_path = model_path
            self.cnn_model.history = None

            # load model if already trained, else train & save it
            if os.path.exists(model_path) and not force_retraining:
                self.cnn_model.load_model()
                print(f"Loaded model {model_path} from disk")
            else:
                print("Model does not exist, training a new one")
                self.cnn_model.build_compile()
                self.cnn_model.train_model_from_numpy(
                    x=train_samples[keep],
                    y=train_labels[keep],
                    val_x=train_samples[~keep],
                    val_y=train_labels[~keep],
                )
                self.cnn_model.save_model()
                print(f"Trained and saved model: {model_path}")

                print("Saving shadow model train history as figure")
                history_fig_path = os.path.join(
                    self.result_path, "sm-training", self.ds.dataset_name
                )
                self.cnn_model.save_train_history(
                    history_fig_path,
                    f"{i}_{self.ds.dataset_name}_shadow_model_training_history.png",
                )

                # test shadow model accuracy
                print("Testing shadow model on test data")
                loss, acc = self.cnn_model.model.evaluate(
                    x=test_samples,
                    y=test_labels,
                    batch_size=self.cnn_model.batch_size,
                    verbose=2,
                )
                print(
                    f"\n============================= DONE TRAINING Shadow Model: {i} =============================\n"
                )

            stat_temp, loss_temp = self.get_stat_and_loss_hinge(
                cnn_model=self.cnn_model, x=train_samples, y=train_labels
            )
            self.stat.append(stat_temp)
            self.losses.append(loss_temp)

            # avoid OOM
            tf.keras.backend.clear_session()
            gc.collect()

        # when recalcuating the stats dont overwrite the indices file
        if not loaded_indices:
            pickle_object(self.in_indices_filename, self.in_indices)

        pickle_object(self.stat_filename, self.stat)
        pickle_object(self.loss_filename, self.losses)

    def attack_shadow_models_amia(self):
        print("Attacking shadow models with AMIA (LIRA)")

        if len(self.stat) == 0 or len(self.losses) == 0:
            print(
                "Error: Before attacking the shadow models with MIA, please train or load the shadow models and retrieve the statistics and losses"
            )
            sys.exit(1)

        if self.attack_result_list is None:
            self.attack_result_list = []

        if self.attack_baseline_result_list is None:
            self.attack_baseline_result_list = []

        slicing_spec = SlicingSpec(entire_dataset=True, by_class=True)

        (train_samples, train_labels) = self.ds.get_train_ds_as_numpy()

        # we currently use the shadow and training models
        for idx in range(self.num_shadow_models + 1):
            print(f"Target model is #{idx}")
            stat_target = self.stat[idx]  # statistics of target model, shape(n,k)
            in_indices_target = self.in_indices[
                idx
            ]  # ground truth membership, shape(n,)

            # `stat_shadow` contains statistics of the shadow models, with shape
            # (num_shadows, n, k).
            stat_shadow = np.array(self.stat[:idx] + self.stat[idx + 1 :])

            # `in_indices_shadow` contains membership of the shadow
            # models, with shape (num_shadows, n). We will use them to get a list
            in_indices_shadow = np.array(
                self.in_indices[:idx] + self.in_indices[idx + 1 :]
            )

            # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
            # (m, k) array, for m being the number of shadow models trained with
            # (resp. without) the j-th example, and k being the number of augmentations
            # (2 in our case).
            stat_in = [
                stat_shadow[:, j][in_indices_shadow[:, j]]
                for j in range(self.num_training_samples)
            ]
            stat_out = [
                stat_shadow[:, j][~in_indices_shadow[:, j]]
                for j in range(self.num_training_samples)
            ]

            # compute the scores and use them for  MIA
            scores = amia.compute_score_lira(
                stat_target, stat_in, stat_out, fix_variance=True
            )

            attack_input = AttackInputData(
                loss_train=scores[in_indices_target],
                loss_test=scores[~in_indices_target],
                labels_train=train_labels[in_indices_target],
                labels_test=train_labels[~in_indices_target],
            )

            result_lira = mia.run_attacks(
                attack_input=attack_input, slicing_spec=slicing_spec
            )
            self.attack_result_list.append(result_lira)

            print("Advanced MIA Attack Results:")
            print(result_lira.calculate_pd_dataframe())

            if self.include_mia:
                # Compare with the baseline MIA using the loss of the target model
                loss_target = self.losses[idx][:, 0]
                attack_input = AttackInputData(
                    loss_train=loss_target[in_indices_target],
                    loss_test=loss_target[~in_indices_target],
                    labels_train=train_labels[in_indices_target],
                    labels_test=train_labels[~in_indices_target],
                )

                result_baseline = mia.run_attacks(
                    attack_input=attack_input, slicing_spec=slicing_spec
                )

                self.attack_baseline_result_list.append(result_baseline)
                print("MIA Attack Results:")
                print(result_baseline.calculate_pd_dataframe())

        # pickle attack result list for LiRA and baseline
        pickle_object(self.attack_result_list_filename, self.attack_result_list)
        pickle_object(
            self.attack_baseline_result_list_filename, self.attack_baseline_result_list
        )

    def get_stat_and_loss_hinge(self, cnn_model: Model, x: np.ndarray, y: np.ndarray):
        # convert labels to integer indexing
        y = tf.argmax(y, axis=1).numpy()

        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            logits = cnn_model.model.predict(x=data, batch_size=cnn_model.batch_size)
            losses.append(utils.log_loss(labels=y, pred=logits, from_logits=True))
            stat.append(
                amia.calculate_statistic(
                    pred=logits, labels=y, is_logits=True, option="hinge"
                )
            )

        return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)

    def get_stat_and_loss_aug_logits(
        self, cnn_model: Model, x: np.ndarray, y: np.ndarray
    ):
        # convert labels to integer indexing
        y = tf.argmax(y, axis=1).numpy()

        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            prob = amia.convert_logit_to_prob(
                cnn_model.model.predict(x=data, batch_size=cnn_model.batch_size)
            )
            losses.append(utils.log_loss(labels=y, pred=prob, from_logits=False))
            stat.append(
                amia.calculate_statistic(
                    pred=prob, labels=y, option="logit", is_logits=False
                )
            )

        return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)

    def _plot_curve_with_area(self, x, y, xlabel, ylabel, ax, label, title=None):
        ax.plot([0, 1], [0, 1], "k-", lw=1.0)
        ax.plot(x, y, lw=2, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text(title)
