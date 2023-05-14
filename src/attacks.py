import tensorflow as tf
from typing import Optional
from cnn_small_model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder, visualize_training
import numpy as np
import pandas as pd

from util import pickle_object, unpickle_object, find_nearest

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests import utils
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults

import matplotlib.pyplot as plt

import functools
from os import sys
import os
import gc


class AmiaAttack():
    """Implementation for multi class advanced mia attack.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used

    Code mostly copied from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py
    """

    def __init__(self,
                 model: CNNModel,
                 ds: AbstractDataset,
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
        self.models_dir = shadow_model_dir
        check_create_folder(self.models_dir)

        self.result_path = result_path
        check_create_folder(self.result_path)

        self.cnn_model: CNNModel = model

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

        self.in_indices: Optional[list] = None  # a list of in-training indices for all models
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

        self.attack_statistics_folder: str = os.path.join(self.result_path, "attack-statistics")
        check_create_folder(self.attack_statistics_folder)
        self.single_model_attack_img_folder: str = os.path.join(self.result_path, "shadow-model-attacks")
        check_create_folder(self.single_model_attack_img_folder)

        self.attack_result_list_filename = os.path.join(self.attack_statistics_folder, f"{ds.dataset_name}_attack_results.pckl")
        self.attack_baseline_result_list_filename = os.path.join(self.attack_statistics_folder, f"{ds.dataset_name}_attack_baseline_results.pckl")

    def load_saved_values(self):
        self.in_indices_filename = unpickle_object(self.in_indices_filename)
        self.stat = unpickle_object(self.stat_filename)
        self.losses = unpickle_object(self.loss_filename)
        self.attack_result_list = unpickle_object(self.attack_result_list_filename)
        self.attack_baseline_result_list = unpickle_object(self.attack_baseline_result_list_filename)

    def calculate_tpr_at_fixed_fpr(self):
        attack_result_frame = pd.DataFrame(columns=["slice feature", "slice value", "train size", "test size", "attack type", "Attacker advantage", "Positive predictive value", "AUC", "fpr@0.1", "fpr@0.001"])

        if self.attack_result_list is None:
            print("Attack result list is None -> cannot proceed to calculate TPR at fixed FPR!")
            return

        for (i, val) in enumerate(self.attack_result_list):
            results: AttackResults = val
            single_frame = results.calculate_pd_dataframe().to_dict("index")[0]  # split dataframe to indexed dict and add it alter to the dataframe again

            single_result = results.single_attack_results[0]
            (idx, _) = find_nearest(single_result.roc_curve.fpr, 0.001)
            fpr_at_001 = single_result.roc_curve.tpr[idx]

            (idx, _) = find_nearest(single_result.roc_curve.fpr, 0.1)
            fpr_at_01 = single_result.roc_curve.tpr[idx]

            single_frame["fpr@0.1"] = fpr_at_01
            single_frame["fpr@0.001"] = fpr_at_001

            attack_result_frame.loc[i] = single_frame

        attack_result_frame.loc["mean"] = attack_result_frame.mean(numeric_only=True)
        attack_result_frame.loc["min"] = attack_result_frame.min(numeric_only=True)
        attack_result_frame.loc["max"] = attack_result_frame.max(numeric_only=True)
        attack_result_frame.loc["var"] = attack_result_frame.var(numeric_only=True)

        print(attack_result_frame)

        df_filename = os.path.join(self.attack_statistics_folder, f"attack_statistic_results_{self.ds.dataset_name}.csv")
        print(f"Saving dataframe as csv: {df_filename}")
        attack_result_frame.to_csv(path_or_buf=df_filename, header=True, index=True, sep="\t")

    def save_all_in_one_roc_curve(self):
        single_results = [x.single_attack_results[0] for x in self.attack_result_list]
        single_results.sort(key=lambda x: x.get_auc(), reverse=True)

        print("Generating all in one ROC curve plot")
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        for res, title in zip(single_results,
                              range(len(self.attack_result_list))):
            label = f'Model #{title} auc={res.get_auc():.4f}'
            plotting.plot_roc_curve(
                res.roc_curve,
                functools.partial(self._plot_curve_with_area, ax=ax, label=label))
        plt.legend()
        plt_name = os.path.join(self.result_path, f"all_in_one_{self.ds.dataset_name}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()

    def save_average_roc_curve(self, generate_all_rocs: bool = True, generate_std_area: bool = True):
        print("Generating average ROC curve plot from all shadow models")

        single_results = [x.single_attack_results[0] for x in self.attack_result_list]
        fpr_len = len(single_results[0].roc_curve.fpr)

        fprs = [i.roc_curve.fpr for i in single_results]
        tprs = [i.roc_curve.tpr for i in single_results]

        fpr_grid = np.logspace(-5, 0, num=fpr_len)

        mean_tpr = np.zeros_like(fpr_grid)

        tpr_int = []

        _, ax = plt.subplots(1, 1, figsize=(10, 10))

        for (fpr, tpr) in zip(fprs, tprs):
            tpr_int.append(np.interp(fpr_grid, fpr, tpr))
            if generate_all_rocs:
                plt.plot(fpr, tpr, 'b', alpha=0.15)

        tpr_int = np.array(tpr_int)
        mean_tpr = tpr_int.mean(axis=0)

        if generate_std_area:
            std_tpr = tpr_int.std(axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = mean_tpr - std_tpr
            plt.fill_between(fpr_grid, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        ax.plot([0, 1], [0, 1], 'r--', lw=1.0)
        ax.plot(fpr_grid, mean_tpr, "b", lw=2, label="Average ROC")
        ax.set(xlabel="TPR", ylabel="FPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text("Receiver Operator Characteristics")

        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()
        plt_name = os.path.join(self.result_path, f"averaged_roc_curve_{self.ds.dataset_name}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()
        return

    def train_load_shadow_models(self, force_retraining: bool = False, force_recalculation: bool = False):
        """Trains, or if shadow models are already trained and saved, loads shadow models from filesystem.

        After training/ loading the shadow models statistics and losses are calulcated over all shadow models.

        """
        (train_samples, train_labels) = self.ds.get_train_ds_as_numpy()
        self.num_training_samples = len(train_samples)

        if not force_recalculation:
            # dont load stats if recalcuation is forced
            self.stat = unpickle_object(self.stat_filename)
            self.losses = unpickle_object(self.loss_filename)

        self.in_indices = unpickle_object(self.in_indices_filename)

        if self.in_indices is not None and self.stat is not None and self.losses is not None:
            print("Loaded in_indices file, stat file and loss file, do not need to load models.")
            return
        else:
            # initialize lists if not loaded before
            self.stat = []
            self.losses = []

        if self.in_indices is None:
            self.in_indices = []

        for i in range(self.num_shadow_models + 1):
            print(f"Creating shadow model {i} and its statistics")

            model_path = os.path.join(self.models_dir,
                                      f"shadow_model_{i}_lr{self.cnn_model.learning_rate}_b{self.cnn_model.batch_size}_e{self.cnn_model.epochs}")

            if not force_recalculation:
                # Generate a binary array indicating which example to include for training
                keep: np.ndarray = np.random.binomial(1, 0.5, size=self.num_training_samples).astype(bool)
                self.in_indices.append(keep)
            else:
                keep = self.in_indices[i]

            # prepare model for new train iteration
            self.cnn_model.model_path = model_path
            self.cnn_model.history = None

            train_count = keep.sum()

            print(
                f"Using {train_count} training samples")

            # load model if already trained, else train & save it
            if os.path.exists(model_path) and not force_retraining:
                self.cnn_model.load_model()
                print(f"Loaded model {model_path} from disk")
            else:
                print("Model does not exist, training a new one")
                self.cnn_model.build_compile()
                self.cnn_model.train_model_from_numpy(x=train_samples[keep],
                                                      y=train_labels[keep],
                                                      val_x=train_samples[~keep],
                                                      val_y=train_labels[~keep],
                                                      batch=self.cnn_model.batch_size)  # specify batch size here, since numpy data is unbatched
                self.cnn_model.save_model()
                print(f"Trained and saved model: {model_path}")

                print("Saving shadow model train history as figure")
                history = self.cnn_model.get_history()

                history_fig_path = os.path.join(self.result_path, "sm-training", self.ds.dataset_name)
                check_create_folder(history_fig_path)

                visualize_training(history=history, img_name=os.path.join(history_fig_path, f"{i}_{self.ds.dataset_name}_shadow_model_training_history.png"))

                # test shadow model accuracy
                print("Testing shadow model on test data")
                self.cnn_model.test_model(self.ds.ds_test)
                print(f"\n============================= DONE TRAINING Shadow Model: {i} =============================\n")

            stat_temp, loss_temp = self.get_stat_and_loss_hinge(
                cnn_model=self.cnn_model,
                x=train_samples,
                y=train_labels)
            self.stat.append(stat_temp)
            self.losses.append(loss_temp)

            # avoid OOM
            tf.keras.backend.clear_session()
            gc.collect()

        # when recalcuating the stats dont overwrite the indices file
        if not force_recalculation:
            pickle_object(self.in_indices_filename, self.in_indices)

        pickle_object(self.stat_filename, self.stat)
        pickle_object(self.loss_filename, self.losses)

    def attack_shadow_models_mia(self):
        print("Attacking shadow models with MIA")

        if len(self.stat) == 0 or len(self.losses) == 0:
            print("Error: Before attacking the shadow models with MIA, please train or load the shadow models and retrieve the statistics and losses")
            sys.exit(1)

        target_model_result_data = pd.DataFrame()
        target_model_result_data_baseline = pd.DataFrame()

        if self.attack_result_list is None:
            self.attack_result_list = []

        if self.attack_baseline_result_list is None:
            self.attack_baseline_result_list = []

        # we currently use the shadow and training models
        for idx in range(self.num_shadow_models + 1):
            print(f"Target model is #{idx}")
            stat_target = self.stat[idx]  # statistics of target model, shape(n,k)
            in_indices_target = self.in_indices[idx]  # ground truth membership, shape(n,)

            # `stat_shadow` contains statistics of the shadow models, with shape
            # (num_shadows, n, k).
            stat_shadow = np.array(self.stat[:idx] + self.stat[idx + 1:])

            # `in_indices_shadow` contains membership of the shadow
            # models, with shape (num_shadows, n). We will use them to get a list
            in_indices_shadow = np.array(self.in_indices[:idx] + self.in_indices[idx + 1:])

            # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
            # (m, k) array, for m being the number of shadow models trained with
            # (resp. without) the j-th example, and k being the number of augmentations
            # (2 in our case).
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
            self.attack_result_list.append(result_lira)
            result_lira_single = result_lira.single_attack_results[0]

            print("Advanced MIA attack with Gaussian:",
                  f"auc = {result_lira_single.get_auc():.4f}",
                  f"adv = {result_lira_single.get_attacker_advantage():.4f}")
            target_model_result_data = pd.concat([target_model_result_data, result_lira.calculate_pd_dataframe()])

            # Compare with the baseline MIA using the loss of the target model
            loss_target = self.losses[idx][:, 0]
            attack_input = AttackInputData(
                loss_train=loss_target[in_indices_target],
                loss_test=loss_target[~in_indices_target])
            result_baseline = mia.run_attacks(attack_input)
            result_baseline_single = result_baseline.single_attack_results[0]
            print('Baseline MIA attack:',
                  f'auc = {result_baseline_single.get_auc():.4f}',
                  f'adv = {result_baseline_single.get_attacker_advantage():.4f}')
            target_model_result_data_baseline = pd.concat([target_model_result_data_baseline, result_baseline.calculate_pd_dataframe()])

            print(f"Generating AUC curve plot for target model {idx}")
            # Plot and save the AUC curves for the three methods.
            _, ax = plt.subplots(1, 1, figsize=(10, 10))
            for res, title in zip([result_lira_single, result_baseline_single],
                                  ['LiRA', 'MIA Baseline (Threshold Attack)']):
                label = f'{title} auc={res.get_auc():.4f}'
                plotting.plot_roc_curve(
                    res.roc_curve,
                    functools.partial(self._plot_curve_with_area, ax=ax, label=label))
            plt.legend()
            plt_name = os.path.join(self.single_model_attack_img_folder, f"model_{self.ds.dataset_name}_id{idx}_advanced_mia.png")
            plt.savefig(plt_name)
            plt.close()

        print("Lira Score results:")
        print(target_model_result_data)

        print("Baseline Score results:")
        print(target_model_result_data_baseline)

        # pickle attack result list for LiRA and baseline
        pickle_object(self.attack_result_list_filename, self.attack_result_list)
        pickle_object(self.attack_baseline_result_list_filename, self.attack_baseline_result_list)

    def get_stat_and_loss_hinge(self,
                                cnn_model: CNNModel,
                                x: np.ndarray,
                                y: np.ndarray,
                                sample_weight: Optional[np.ndarray] = None):
        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            logits = cnn_model.model.predict(x=data, batch_size=cnn_model.batch_size)
            losses.append(utils.log_loss(labels=y,
                                         pred=logits,
                                         from_logits=True,
                                         sample_weight=sample_weight))
            stat.append(
                amia.calculate_statistic(
                    pred=logits,
                    labels=y,
                    is_logits=True,
                    option="hinge",
                    sample_weight=sample_weight))

        return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)

    def get_stat_and_loss_aug_logits(self,
                                     cnn_model: CNNModel,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     sample_weight: Optional[np.ndarray] = None):
        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            prob = amia.convert_logit_to_prob(
                cnn_model.model.predict(x=data,
                                        batch_size=cnn_model.batch_size))
            losses.append(utils.log_loss(labels=y,
                                         pred=prob,
                                         from_logits=False,
                                         sample_weight=sample_weight))
            stat.append(
                amia.calculate_statistic(
                    pred=prob,
                    labels=y,
                    option="logit",
                    sample_weight=sample_weight,
                    is_logits=False))

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
