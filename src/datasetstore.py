import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults, SingleAttackResult
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import unpickle_object, find_nearest, plot_curve_with_area
from ppml_datasets.utils import check_create_folder

import functools
import os
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
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
    attack_result_folder: str

    # data to be stored
    in_indices: List[np.ndarray] = field(default_factory=list)
    stat_list: List[np.ndarray] = field(default_factory=list)
    losses_list: List[np.ndarray] = field(default_factory=list)
    attack_result_list: List[AttackResults] = field(default_factory=list)
    attack_baseline_result_list: List[AttackResults] = field(default_factory=list)

    attack_result_df: pd.DataFrame = None
    attack_baseline_result_df: pd.DataFrame = None
    attack_result_image_path: str = None
    attack_result_image_path_mia_vs_amia_path: str = None

    mean_tpr = None
    mean_fpr = None

    best_idx_fpr0001 = None
    best_idx_fpr01 = None
    best_idx_auc = None

    def __post_init__(self):
        """Auto-Initialize values dependant from other values."""
        if self.attack_result_folder is None:
            self.attack_result_folder = os.path.dirname(self.attack_result_list_filename)

        if self.attack_result_image_path is None:
            self.attack_result_image_path = os.path.join(self.attack_result_folder, "figs", f"{self.ds_name}")
            check_create_folder(self.attack_result_image_path)

        if self.attack_result_image_path_mia_vs_amia_path is None:
            self.attack_result_image_path_mia_vs_amia_path = os.path.join(self.attack_result_image_path, "mia-vs-amia")
            check_create_folder(self.attack_result_image_path_mia_vs_amia_path)

    def load_saved_values(self):
        self.in_indices_filename = unpickle_object(self.in_indices_filename)
        self.stat = unpickle_object(self.stat_filename)
        self.losses = unpickle_object(self.loss_filename)
        self.attack_result_list = unpickle_object(self.attack_result_list_filename)
        self.attack_baseline_result_list = unpickle_object(self.attack_baseline_result_list_filename)

    def get_fpr_at_fixed_tpr(self, single_attack_result: SingleAttackResult) -> Tuple[float, float]:
        """Caclulate FPR @ (0.1, 0.001) TPR.

        Return:
        ------
        Tuple[float, float] -> (fpr_at_01, fpr_at_001)

        """
        (idx, _) = find_nearest(single_attack_result.roc_curve.fpr, 0.001)
        fpr_at_001 = single_attack_result.roc_curve.tpr[idx]

        (idx, _) = find_nearest(single_attack_result.roc_curve.fpr, 0.1)
        fpr_at_01 = single_attack_result.roc_curve.tpr[idx]

        return (fpr_at_01, fpr_at_001)

    def create_complete_dataframe(self, attack_result_list: List[AttackResults], attack_name: str) -> pd.DataFrame:
        attack_result_frame = pd.DataFrame(columns=["slice feature", "slice value", "train size", "test size", "attack type", "Attacker advantage", "Positive predictive value", "AUC", "fpr@0.1", "fpr@0.001"])

        if attack_result_list is None:
            print("Attack result list is None -> cannot proceed to calculate TPR at fixed FPR!")
            return

        for (i, val) in enumerate(attack_result_list):
            results: AttackResults = val
            single_frame = results.calculate_pd_dataframe().to_dict("index")[0]  # split dataframe to indexed dict and add it alter to the dataframe again

            fpr_at_01, fpr_at_001 = self.get_fpr_at_fixed_tpr(results.single_attack_results[0])

            single_frame["fpr@0.1"] = fpr_at_01
            single_frame["fpr@0.001"] = fpr_at_001

            attack_result_frame.loc[i] = single_frame

        mean = attack_result_frame.mean(numeric_only=True)
        min = attack_result_frame.min(numeric_only=True)
        max = attack_result_frame.max(numeric_only=True)
        var = attack_result_frame.var(numeric_only=True)

        attack_result_frame.loc["mean"] = mean
        attack_result_frame.loc["min"] = min
        attack_result_frame.loc["max"] = max
        attack_result_frame.loc["var"] = var

        print(attack_result_frame)

        return attack_result_frame

    def create_mia_vs_amia_roc_curves(self):
        print("Generating AUC curve plot for comparing amia and mia")

        for idx, (result_lira, result_baseline) in enumerate(zip(self.attack_result_list, self.attack_baseline_result_list)):
            result_lira_single: SingleAttackResult = result_lira.single_attack_results[0]
            result_baseline_single: SingleAttackResult = result_baseline.single_attack_results[0]
            # Plot and save the AUC curves for the three methods.
            _, ax = plt.subplots(1, 1, figsize=(10, 10))
            for res, title in zip([result_lira_single, result_baseline_single],
                                  ['LiRA', 'MIA Baseline (Threshold Attack)']):
                label = f'{title} auc={res.get_auc():.4f}'
                plotting.plot_roc_curve(
                    res.roc_curve,
                    functools.partial(plot_curve_with_area, ax=ax, label=label, use_log_scale=True, title=f"MIA vs Advanced MIA (LiRA) - Log scale #{idx}"))
            plt.legend()
            plt_name = os.path.join(self.attack_result_image_path_mia_vs_amia_path, f"model_{self.ds_name}_id{idx}_log_scaled_advanced_mia.png")
            print(f"Saving MIA vs AMIA {plt_name}")
            plt.savefig(plt_name)
            plt.close()

            _, ax = plt.subplots(1, 1, figsize=(10, 10))
            for res, title in zip([result_lira_single, result_baseline_single],
                                  ['LiRA', 'MIA Baseline (Threshold Attack)']):
                label = f'{title} auc={res.get_auc():.4f}'
                plotting.plot_roc_curve(
                    res.roc_curve,
                    functools.partial(plot_curve_with_area, ax=ax, label=label, use_log_scale=False, title=f"MIA vs Advanced MIA (LiRA) - Linear scale #{idx}"))
            plt.legend()
            plt_name = os.path.join(self.attack_result_image_path_mia_vs_amia_path, f"model_{self.ds_name}_id{idx}_linear_scaled_advanced_mia.png")
            print(f"Saving MIA vs AMIA {plt_name}")
            plt.savefig(plt_name)
            plt.close()

    def create_all_in_one_roc_curve(self):
        single_results = [x.single_attack_results[0] for x in self.attack_result_list]
        single_results.sort(key=lambda x: x.get_auc(), reverse=True)

        print("Generating all in one ROC curve plot")
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        for res, title in zip(single_results,
                              range(len(self.attack_result_list))):
            label = f'Model #{title} auc={res.get_auc():.4f}'
            plotting.plot_roc_curve(
                res.roc_curve,
                functools.partial(plot_curve_with_area, ax=ax, label=label, use_log_scale=True, title="All Shadow-Model's ROC Curves"))
        plt.legend()
        plt_name = os.path.join(self.attack_result_image_path, f"all_curves_{self.ds_name}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()

    def create_average_roc_curve(self, attack_result_list: List[AttackResults], name: str = "AMIA", generate_all_rocs: bool = True, generate_std_area: bool = True) -> Tuple[np.array, np.array]:
        """Create average ROC curve from attack data generated from the shadow models.

        Return:
        ------
        Tuple[np.array, np.array] -> (mean_tpr, aligned FPR)

        """
        print("Generating average ROC curve plot from all shadow models")

        single_results = [x.single_attack_results[0] for x in attack_result_list]
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
        ax.title.set_text(f"Receiver Operator Characteristics - {name}")

        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        options: str = ""
        if generate_all_rocs:
            options += "_all_rocs_"
        if generate_std_area:
            options += "_std_bounds_"

        plt_name = os.path.join(self.attack_result_image_path, f"averaged_roc_curve_{self.ds_name}{options}_{name}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()
        return (mean_tpr, fpr_grid)

    def set_best_attack_run_idx(self, attack_result: pd.DataFrame):
        a = attack_result.idxmax(axis=0, skipna=True)
        self.best_idx_fpr0001 = a["fpr@0.001"]
        self.best_idx_fpr01 = a["fpr@0.1"]
        self.best_idx_auc = a["AUC"]
