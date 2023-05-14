import tensorflow as tf
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import pickle_object, unpickle_object, find_nearest
from cnn_small_model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder

import functools
import os
from typing import Optional, List, Dict
from dataclasses import dataclass


class Analyser():
    def __init__(self,
                 ds_list: List[AbstractDataset],
                 run_number: int,
                 result_path: str,
                 model_path: str,
                 num_shadow_models: int,
                 ):
        self.result_path = result_path
        check_create_folder(self.result_path)

        self.ds_list = ds_list
        self.num_shadow_models = num_shadow_models

        self.amia_result_path = os.path.join(result_path, str(run_number))
        self.attack_statistics_folder: str = os.path.join(self.result_path, "attack-statistics")
        self.single_model_train_results: str = os.path.join(self.result_path, "single-model-train")

        self.dataset_data: Dict[str, DatasetStore] = {}

        for ds in ds_list:
            model_save_path: str = os.path.join(model_path, str(run_number), ds.dataset_name)
            shadow_model_save_path: str = os.path.join(model_path, str(run_number), "shadow_models", ds.dataset_name)
            numpy_path: str = os.path.join(shadow_model_save_path, "data")
            in_indices_filename = os.path.join(numpy_path, "in_indices.pckl")
            stat_filename = os.path.join(numpy_path, "model_stat.pckl")
            loss_filename = os.path.join(numpy_path, "model_loss.pckl")

            attack_result_list_filename = os.path.join(self.attack_statistics_folder, f"{ds.dataset_name}_attack_results.pckl")
            attack_baseline_result_list_filename = os.path.join(self.attack_statistics_folder, f"{ds.dataset_name}_attack_baseline_results.pckl")

            ds_store = DatasetStore(shadow_model_dir=shadow_model_save_path,
                                    model_save_path=model_save_path,
                                    ds_name=ds.dataset_name,
                                    numpy_path=numpy_path,
                                    in_indices_filename=in_indices_filename,
                                    stat_filename=stat_filename,
                                    loss_filename=loss_filename,
                                    attack_result_list_filename=attack_result_list_filename,
                                    attack_baseline_result_list=attack_baseline_result_list_filename,
                                    )
            self.dataset_data[ds.dataset_name] = ds_store

        print(self.dataset_data)


@dataclass(init=False)
class DatasetStore():
    """Represents all kind of data related to a dataset + shadow model results."""

    # filenames to be stored
    shadow_model_dir: str  # path to the shadow models and numpy files
    model_save_path: str  # path of weights to the single-trained models
    ds_name: str
    numpy_path: str
    in_indices_filename: str
    stat_filename: str
    loss_filename: str
    attack_result_list_filename: str
    attack_baseline_result_list_filename: str

    # data to be stored
    in_indices: List[np.ndarray]
    stat_list: List[np.ndarray]
    losses_list: List[np.ndarray]
    attack_result_list: List[AttackResults]
    attack_baseline_result_list: List[AttackResults]

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


def _plot_curve_with_area(x, y, xlabel, ylabel, ax, label, title=None):
    ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
    ax.plot(x, y, lw=2, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set(aspect=1, xscale='log', yscale='log')
    ax.title.set_text(title)
