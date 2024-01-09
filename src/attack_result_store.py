import functools
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
    AttackResults,
    SingleAttackResult,
    SlicingFeature,
)

from util import find_nearest, plot_curve_with_area, save_dataframe, unpickle_object


class AttackType(Enum):
    LIRA = "lira"
    MIA = "mia"


FIGSIZE = (5, 5)


@dataclass
class AttackResultStore:
    """Stores all attack result data for an attack."""

    # filenames to be stored
    attack_type: AttackType

    ds_name: str
    run_number: int

    # the path until attack-statistics or attack-analysis folder
    base_path: str

    attack_result_list: List[AttackResults] = field(default_factory=list)
    attack_result_df: pd.DataFrame = None
    fpr_grid: np.ndarray = None
    mean_tpr: np.ndarray = None

    def get_statistics_folder(self) -> str:
        return os.path.join(self.base_path, str(self.run_number), "attack-statistics")

    def get_analysis_folder(self) -> str:
        return os.path.join(
            self.base_path,
            str(self.run_number),
            "attack-analysis",
            self.ds_name,
        )

    def load_saved_values(self):
        attack_result_list_filename = os.path.join(
            self.get_statistics_folder(),
            f"{self.ds_name}_attack_{self.attack_type.value}_results.pckl",
        )
        self.attack_result_list = unpickle_object(attack_result_list_filename)

    def create_complete_dataframe(self) -> pd.DataFrame:
        """Combine the dataframes from all attacked models. Also adds mean, min, max calculation results for every attack slice type (Entire Dataset, Class=X)."""
        columns = [
            "attack type",
            "shadow model",
            "slice feature",
            "slice value",
            "train size",
            "test size",
            "Attacker advantage",
            "Positive predictive value",
            "AUC",
            "fpr@0.1",
            "fpr@0.001",
        ]
        attack_result_frame = pd.DataFrame(columns=columns)

        if self.attack_result_list is None:
            print(
                "Attack result list is None -> cannot proceed to calculate TPR at fixed FPR!"
            )
            return

        slice_spec_list = []

        for result in self.attack_result_list[0].single_attack_results:
            slice_spec_list.append(str(result.slice_spec))

        for i, val in enumerate(self.attack_result_list):
            results: AttackResults = val
            # split dataframe to indexed dict and add it alter to the dataframe again
            df = results.calculate_pd_dataframe()

            df["attack type"] = self.attack_type.value
            df["shadow model"] = i

            fpr_at_01_list = []
            fpr_at_001_list = []
            for single_result in results.single_attack_results:
                fpr_at_01, fpr_at_001 = self.get_fpr_at_fixed_tpr(single_result)

                fpr_at_01_list.append(fpr_at_01)
                fpr_at_001_list.append(fpr_at_001)

            df["fpr@0.1"] = fpr_at_01_list
            df["fpr@0.001"] = fpr_at_001_list

            df.reset_index(inplace=True)
            df = df.reindex(columns, axis=1)

            attack_result_frame = pd.concat(
                [attack_result_frame, df], ignore_index=True
            )

        group = attack_result_frame.groupby(["slice feature", "slice value"])[
            [
                "Attacker advantage",
                "Positive predictive value",
                "AUC",
                "fpr@0.1",
                "fpr@0.001",
            ]
        ]
        mean_df = group.mean(numeric_only=True)
        max_df = group.max(numeric_only=True)
        std_df = group.std(numeric_only=True)

        # set nice index values
        idx_mean = pd.Index([f"mean {spec}" for spec in slice_spec_list])
        mean_df.set_index(idx_mean, inplace=True)

        idx_max = pd.Index([f"max {spec}" for spec in slice_spec_list])
        max_df.set_index(idx_max, inplace=True)

        idx_std = pd.Index([f"std {spec}" for spec in slice_spec_list])
        std_df.set_index(idx_std, inplace=True)

        mean_df["attack type"] = self.attack_type.value
        max_df["attack type"] = self.attack_type.value
        std_df["attack type"] = self.attack_type.value

        attack_result_df = pd.concat(
            [attack_result_frame, mean_df, std_df, max_df], ignore_index=False
        ).round(decimals=4)

        df_filename = os.path.join(
            self.get_analysis_folder(),
            f"{self.attack_type.value}_attack_statistic_results_{self.ds_name}.csv",
        )
        save_dataframe(attack_result_df, df_filename)

        self.attack_result_df = attack_result_df

        return attack_result_df

    def get_fpr_at_fixed_tpr(
        self, single_attack_result: SingleAttackResult
    ) -> Tuple[float, float]:
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

    def create_entire_dataset_combined_roc_curve(self):
        """Print the attack results on Entire Dataset Slice of every shadow model into a single figure."""
        entire_ds_attack_list = self.get_single_entire_ds_attack_results()
        entire_ds_attack_list.sort(key=lambda x: x.get_auc(), reverse=True)

        print("Generating all in one ROC curve plot")
        _, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        for res, title in zip(
            entire_ds_attack_list, range(len(self.attack_result_list))
        ):
            label = f"Model #{title} auc={res.get_auc():.3f}"
            plotting.plot_roc_curve(
                res.roc_curve,
                functools.partial(
                    plot_curve_with_area,
                    ax=ax,
                    label=label,
                    use_log_scale=True,
                    title=None,
                ),
            )
        plt.legend()
        plt_name = os.path.join(
            self.get_analysis_folder(),
            f"{self.ds_name}_{self.attack_type.value}_all_roc_entire_ds_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()

    def get_single_entire_ds_attack_results(self) -> List[SingleAttackResult]:
        result_list = []
        for attack_results in self.attack_result_list:
            result_list.append(self._extract_entire_dataset_slice(attack_results))
        return result_list

    def get_single_class_attack_result_dict(
        self,
    ) -> Dict[str, List[SingleAttackResult]]:
        single_attack_dict = defaultdict(list)

        # these are the attack_results for every shadow model
        for attack_results in self.attack_result_list:
            for result in attack_results.single_attack_results:
                if result.slice_spec.feature is SlicingFeature.CLASS:
                    class_number = str(result.slice_spec.value)
                    single_attack_dict[class_number].append(result)

        return single_attack_dict

    def create_average_class_attack_roc(self):
        """Create an averaged ROC curve over all class attack slices.

        Creating a ROC curve with N curves, where N is the number of classes.
        """
        class_attack_dict = self.get_single_class_attack_result_dict()

        # get item from dict to get fpr_len
        fpr_len = len(next(iter(class_attack_dict.values()))[0].roc_curve.fpr)
        fpr_grid = np.logspace(-5, 0, num=fpr_len)

        class_wise_mean_tpr_dict = {}

        for k, v in class_attack_dict.items():
            mean_tpr, _, grid_fpr = self.calculate_mean_tpr_and_fpr(
                v, fpr_grid=fpr_grid
            )
            class_wise_mean_tpr_dict[k] = mean_tpr

        _, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        for k, v in class_wise_mean_tpr_dict.items():
            ax.plot(fpr_grid, v, lw=2, label=f"Class {k}")

        ax.set(xlabel="TPR", ylabel="FPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.get_analysis_folder(),
            f"{self.ds_name}_{self.attack_type.value}_all_roc_class_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved averaged ROC curve {plt_name} (class wise)")
        plt.close()

    def calculate_mean_tpr_and_fpr(
        self,
        results: List[SingleAttackResult],
        fpr_grid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fpr_grid is None:
            fpr_len = len(results[0].roc_curve.fpr)
            fpr_grid = np.logspace(-5, 0, num=fpr_len)

        # get all fpr and tpr values from single class slice
        fprs = [i.roc_curve.fpr for i in results]
        tprs = [i.roc_curve.tpr for i in results]

        tpr_int = []

        for fpr, tpr in zip(fprs, tprs):
            tpr_int.append(np.interp(fpr_grid, fpr, tpr))

        tpr_int = np.array(tpr_int)
        mean_tpr = tpr_int.mean(axis=0)
        return (mean_tpr, tpr_int, fpr_grid)

    def create_average_roc_curve_entire_dataset(
        self,
        generate_std_area: bool = True,
    ) -> Tuple[np.array, np.array]:
        """Create average ROC curve from attack data generated from the shadow models.

        Return:
        ------
        Tuple[np.array, np.array] -> (mean_tpr, aligned FPR)

        """
        entire_ds_results = self.get_single_entire_ds_attack_results()

        mean_tpr, tpr_int, fpr_grid = self.calculate_mean_tpr_and_fpr(entire_ds_results)

        self.mean_tpr = mean_tpr
        self.fpr_grid = fpr_grid
        avg_auc = self.attack_result_df.loc["mean Entire dataset"]["AUC"]
        fpr0001 = self.attack_result_df.loc["mean Entire dataset"]["fpr@0.001"]
        fpr01 = self.attack_result_df.loc["mean Entire dataset"]["fpr@0.1"]

        _, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        if generate_std_area:
            std_tpr = tpr_int.std(axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = mean_tpr - std_tpr
            plt.fill_between(fpr_grid, tprs_lower, tprs_upper, color="grey", alpha=0.3)

        ax.plot([0, 1], [0, 1], "k--", lw=1.0)
        ax.plot(
            fpr_grid,
            mean_tpr,
            "b",
            lw=2,
            label=f"AUC={avg_auc:.3f}\nTPR@0.001={fpr0001:.3f}\nTPR@0.1={fpr01:.3f}",
        )
        ax.set(xlabel="TPR", ylabel="FPR")
        ax.set(aspect=1, xscale="log", yscale="log")

        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend()

        options: str = ""
        if generate_std_area:
            options += "_std_bounds_"

        plt_name = os.path.join(
            self.get_analysis_folder(),
            f"{self.ds_name}_{self.attack_type.value}{options}average_roc_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved averaged ROC curve {plt_name} (entire dataset)")
        plt.close()
        return (mean_tpr, fpr_grid)

    def get_attack_df_entire_dataset_only(self) -> pd.DataFrame:
        """Return a dataframe, which only represents the Entire dataset attack slice."""
        return self.attack_result_df[
            self.attack_result_df["slice feature"] == "Entire dataset"
        ]

    def _extract_entire_dataset_slice(
        self, result: AttackResults
    ) -> SingleAttackResult:
        """Extract the Entire Dataset slice from an AttackResult object."""
        for i in result.single_attack_results:
            if i.slice_spec.feature is None:
                return i

    def get_best_fpr001_run(self) -> SingleAttackResult:
        return self._get_best_X_from_run("fpr@0.001")

    def get_best_fpr01_run(self) -> SingleAttackResult:
        return self._get_best_X_from_run("fpr@0.1")

    def get_best_auc_run(self) -> SingleAttackResult:
        return self._get_best_X_from_run("AUC")

    def _get_best_X_from_run(self, col_name: str) -> SingleAttackResult:
        # get ID of best Entire dataset X (fpr@0.001, auc, ...)
        best_idx = self.get_attack_df_entire_dataset_only()[col_name].idxmax()

        shadow_model_number = self.attack_result_df.loc[best_idx]["shadow model"]

        attack_results = self.attack_result_list[shadow_model_number]
        return self._extract_entire_dataset_slice(attack_results)
