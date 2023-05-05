import tensorflow as tf
from typing import Optional
from model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder, visualize_training
import numpy as np
import pandas as pd
import pickle


from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests import utils
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia

import matplotlib.pyplot as plt

import functools
from os import sys
import os
import gc
import copy


class AmiaAttack():
    """Implementation for multi class advanced mia attack.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used

    Code mostly copied from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py
    """

    def __init__(self,
                 model: CNNModel,
                 ds: AbstractDataset,
                 run_name: int,
                 result_path: str,
                 shadow_model_dir: str = "data/models/amia/shadow_models",
                 num_shadow_models: int = 16,
                 ):
        """Initialize MiaAttack class.

        Paramter:
        ---------
        model : CNNModel
        dataset : AbstractDataset - When instatiating a Dataset, a validation dataset is not needed, instead increase the train sice when specifying the train_val_test_split
        """
        self.models_dir = os.path.join(shadow_model_dir, run_name, ds.dataset_name)
        check_create_folder(self.models_dir)

        self.result_path = result_path
        check_create_folder(self.result_path)

        self.cnn_model: CNNModel = model
        self.run_name = run_name

        if ds.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            sys.exit(1)

        if ds.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            sys.exit(1)

        self.ds = ds
        if self.ds.num_classes is None:
            self.ds.get_number_of_classes()

        self.num_shadow_models = num_shadow_models

        self.in_indices = []  # a list of in-training indices for all models
        self.stat = []  # a list of statistics for all models
        self.losses = []  # a list of losses for all models
        self.num_training_samples: int = 0

    def train_load_shadow_models(self):
        """Trains, or if shadow models are already trained and saved, loads shadow models from filesystem.

        After training/ loading the shadow models statistics and losses are calulcated over all shadow models.

        """
        (train_samples, train_labels) = self.ds.get_train_ds_as_numpy()
        self.num_training_samples = len(train_samples)

        loaded_indices: bool = False
        # try loading the in_indices if it was saved
        numpy_path = os.path.join(self.models_dir, "data")
        numpy_file = os.path.join(numpy_path, "in_indices.pckl")

        check_create_folder(numpy_path)

        if os.path.isfile(numpy_file):
            with open(numpy_file, "rb") as f:
                self.in_indices = pickle.load(f)
                loaded_indices = True
                print(f"Loaded .pckl file for indices: {numpy_file}")

        for i in range(self.num_shadow_models + 1):
            print(f"Creating shadow model {i}")

            model_path = os.path.join(self.models_dir,
                                      f"r{self.run_name}_shadow_model_{i}_lr{self.cnn_model.learning_rate}_b{self.cnn_model.batch_size}_e{self.cnn_model.epochs}")

            if not loaded_indices:
                # Generate a binary array indicating which example to include for training
                keep: np.ndarray = np.random.binomial(1, 0.5, size=self.num_training_samples).astype(bool)
                self.in_indices.append(keep)
            else:
                keep = self.in_indices[i]

            # we want to create an exact copy of the already trained model, but change model path
            shadow_model: CNNModel = copy.copy(self.cnn_model)
            shadow_model.model_path = model_path
            shadow_model.reset_model_optimizer()

            train_count = keep.sum()

            print(
                f"Using {train_count} training samples")

            # load model if already trained, else train & save it
            if os.path.exists(model_path):
                shadow_model.load_model()
                print(f"Loaded model {model_path} from disk")
            else:
                shadow_model.build_compile()
                shadow_model.train_model_from_numpy(x=train_samples[keep],
                                                    y=train_labels[keep],
                                                    val_x=train_samples[~keep],
                                                    val_y=train_labels[~keep],
                                                    batch=self.cnn_model.batch_size)  # specify batch size here, since numpy data is unbatched
                shadow_model.save_model()
                print(f"Trained and saved model: {model_path}")

                print("Saving shadow model train history as figure")
                history = shadow_model.get_history()

                history_fig_path = os.path.join(self.result_path, "sm-training", self.run_name)
                check_create_folder(history_fig_path)

                visualize_training(history=history, img_name=os.path.join(history_fig_path, f"{i}_{self.ds.dataset_name}_shadow_model_training_history.png"))

                # test shadow model accuracy
                print("Testing shadow model on test data")
                shadow_model.test_model(self.ds.ds_test)
                print(f"\n============================= DONE TRAINING Shadow Model: {i} =============================\n")

            stat_temp, loss_temp = self._get_stat_and_loss(
                cnn_model=shadow_model,
                x=train_samples,
                y=train_labels)
            self.stat.append(stat_temp)
            self.losses.append(loss_temp)

            # avoid OOM
            tf.keras.backend.clear_session()
            gc.collect()

        # save in_indices when not existing
        if not os.path.isfile(numpy_file):
            with open(numpy_file, "wb") as f:
                pickle.dump(self.in_indices, f)
                print(f"Saved .pckl file {numpy_file}")

    def attack_shadow_models_mia(self, plot_auc_curve: bool = True, plot_filename: Optional[str] = "advanced_mia.png"):
        print("Attacking shadow models with MIA")

        if len(self.stat) == 0 or len(self.losses) == 0:
            print("Error: Before attacking the shadow models with MIA, please train or load the shadow models and retrieve the statistics and losses")
            sys.exit(1)

        # pd.set_option("display.max_rows", 8, "display.max_columns", None)
        target_model_result_data = pd.DataFrame()

        # we currently use the shadow and training models
        for idx in range(self.num_shadow_models + 1):
            print(f"Target model is #{idx}")
            stat_target = self.stat[idx]  # statistics of target model, shape(n,k)
            in_indices_target = self.in_indices[idx]  # ground truth membership, shape(n,)

            # `stat_shadow` contains statistics of the shadow models, with shape
            # (num_shadows, n, k). `in_indices_shadow` contains membership of the shadow
            # models, with shape (num_shadows, n). We will use them to get a list
            # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
            # (m, k) array, for m being the number of shadow models trained with
            # (resp. without) the j-th example, and k being the number of augmentations
            # (2 in our case).
            stat_shadow = np.array(self.stat[:idx] + self.stat[idx + 1:])
            in_indices_shadow = np.array(self.in_indices[:idx] + self.in_indices[idx + 1:])
            stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]]
                       for j in range(self.num_training_samples)]
            stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]]
                        for j in range(self.num_training_samples)]

            # compute the scores and use them for  MIA
            scores = amia.compute_score_lira(stat_target, stat_in, stat_out, fix_variance=True)

            attack_input = AttackInputData(
                loss_train=scores[in_indices_target],
                loss_test=scores[~in_indices_target])

            result_lira = mia.run_attacks(attack_input)

            result_lira_single = result_lira.single_attack_results[0]

            print("Advanced MIA attack with Gaussian:",
                  f"auc = {result_lira_single.get_auc():.4f}",
                  f"adv = {result_lira_single.get_attacker_advantage():.4f}")
            target_model_result_data = pd.concat([target_model_result_data, result_lira.calculate_pd_dataframe()])

            if plot_auc_curve:
                print(f"Generating AUC curve plot for target model {idx}")
                # Plot and save the AUC curves for the three methods.
                _, ax = plt.subplots(1, 1, figsize=(5, 5))
                for res, title in zip([result_lira_single],
                                      ['LiRA']):
                    label = f'{title} auc={res.get_auc():.4f}'
                    plotting.plot_roc_curve(
                        res.roc_curve,
                        functools.partial(self._plot_curve_with_area, ax=ax, label=label))
                plt.legend()
                plt_name = os.path.join(self.result_path, f"model_id{idx}_{plot_filename}")
                plt.savefig(plt_name)

        # try:
        #    result_lira.save("result_lira.pckl")
        # except Exception as e:
        #    print(f"could not picke results! {e}")

        print("Lira Score results:")
        print(target_model_result_data)

    def _get_stat_and_loss(self,
                           cnn_model: CNNModel,
                           x: np.ndarray,
                           y: np.ndarray,
                           sample_weight: Optional[np.ndarray] = None):
        """Get the statistics and losses.

        Paramter
        --------
          model: model to make prediction
          x: samples
          y: true labels of samples (integer valued)
          sample_weight: a vector of weights of shape (n_samples, ) that are
            assigned to individual samples. If not provided, then each sample is
            given unit weight. Only the LogisticRegressionAttacker and the
            RandomForestAttacker support sample weights.

        Returns
        -------
          the statistics and cross-entropy losses

        """
        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            prob = amia.convert_logit_to_prob(
                cnn_model.model.predict(data, batch_size=cnn_model.batch_size))
            losses.append(utils.log_loss(labels=y,
                                         pred=prob,
                                         from_logits=False,
                                         sample_weight=sample_weight))
            stat.append(
                amia.calculate_statistic(
                    pred=prob,
                    labels=y,
                    is_logits=False,
                    option="logit",
                    sample_weight=sample_weight))
        return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)

    def _plot_curve_with_area(self, x, y, xlabel, ylabel, ax, label, title=None):
        ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
        ax.plot(x, y, lw=2, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text(title)

# We also try using `compute_score_offset` to compute the score. We take
# the negative of the score, because higher statistics corresponds to higher
# probability for in-training, which is the opposite of loss.
# scores = -amia.compute_score_offset(stat_target, stat_in, stat_out)
# attack_input = AttackInputData(
#     loss_train=scores[in_indices_target],
#     loss_test=scores[~in_indices_target])
# result_offset = mia.run_attacks(attack_input)
# result_offset_single = result_offset.single_attack_results[0]
# print('Advanced MIA attack with offset:',
#       f'auc = {result_offset_single.get_auc():.4f}',
#       f'adv = {result_offset_single.get_attacker_advantage():.4f}')

# Compare with the baseline MIA using the loss of the target model
# loss_target = self.losses[idx][:, 0]
# attack_input = AttackInputData(
#     loss_train=loss_target[in_indices_target],
#     loss_test=loss_target[~in_indices_target])
# result_baseline = mia.run_attacks(attack_input)
# result_baseline_single = result_baseline.single_attack_results[0]
# print('Baseline MIA attack:',
#       f'auc = {result_baseline_single.get_auc():.4f}',
#       f'adv = {result_baseline_single.get_attacker_advantage():.4f}')
