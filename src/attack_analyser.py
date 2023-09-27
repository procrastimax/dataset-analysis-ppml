import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attack_result_store import AttackResultStore, AttackType
from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import get_run_numbers, save_dataframe

pd.options.mode.chained_assignment = None


class AttackAnalyser:

    def __init__(
        self,
        settings: RunSettings,
        result_path: str,
        model_path: str,
    ):
        self.result_path = result_path
        check_create_folder(self.result_path)

        self.settings: RunSettings = settings
        self.model_path: str = model_path

        # base folder where all individual runs are saved
        self.run_result_folder = os.path.join(result_path, settings.model_name,
                                              settings.run_name)

        self.analysis_combined_runs = os.path.join(
            result_path,
            settings.model_name,
            settings.run_name,
            "attack-analysis-combined",
        )

    def get_combined_ds_analysis_folder(self, run_number: int) -> str:
        return os.path.join(
            self.run_result_folder,
            str(run_number),
            "attack-analysis",
            "combined-ds",
        )

    def load_attack_results(
        self,
        attack_type: AttackType,
        run_number: int,
        ds_name_list: List[str],
    ) -> Dict[str, AttackResultStore]:
        attack_result_dict: Dict[str, AttackResultStore] = {}

        for ds_name in ds_name_list:
            attack_results_store = AttackResultStore(
                attack_type=attack_type,
                ds_name=ds_name,
                run_number=run_number,
                base_path=self.run_result_folder,
            )
            check_create_folder(attack_results_store.get_analysis_folder())

            attack_results_store.load_saved_values()
            attack_results_store.create_complete_dataframe()

            attack_result_dict[ds_name] = attack_results_store

        return attack_result_dict

    def compile_attack_results(self, attack_type: AttackType):
        # only use all existant run numbers (by scanning the run name folder), when the anaylsis run parameter is unset
        if self.settings.analysis_run_numbers is None:
            runs = get_run_numbers(self.run_result_folder)
        else:
            runs = self.settings.analysis_run_numbers

        run_ds_name_dict: Dict[int, List[str]] = {}

        for run in runs:
            print(
                f" --- Creating attack analysis for run: {self.settings.run_name} -> {run} ---"
            )
            # load pararmeter.json and ds_name_list
            parameter_file_path = os.path.join(self.run_result_folder,
                                               str(run), "parameter.json")
            ds_list: List[str] = None
            with open(parameter_file_path) as parameter_file:
                parameter_dict = json.load(parameter_file)
                ds_list = parameter_dict["datasets"]
                run_ds_name_dict[run] = ds_list

            if ds_list is None:
                print(
                    "Cannot load attack results, since the parameter.json file does not contain used dataset names for the runs!"
                )
                sys.exit(1)

            result_dict = self.load_attack_results(attack_type=attack_type,
                                                   run_number=run,
                                                   ds_name_list=ds_list)
            self._compile_attack_results(attack_type, result_dict, run)

        # only combine multiple runs if there are any
        if len(runs) > 1:
            print(" --- Creating combined runs graphics ---")
            self.create_runs_combined_graphics(attack_type, runs,
                                               run_ds_name_dict)

    def _compile_attack_results(
        self,
        attack_type: AttackType,
        attack_result_dict: Dict[str, AttackResultStore],
        run_number: int,
    ):
        """Create an overview of compiled attack results for a single run."""
        for ds_name, store in attack_result_dict.items():
            store.create_entire_dataset_combined_roc_curve()

            # this function sets the store's mean_tpr and fpr_grid
            store.create_average_roc_curve_entire_dataset(
                generate_std_area=True)
            store.create_average_class_attack_roc()

        # create figures to compare the best runs of each dataset with each other
        attack_stores = list(attack_result_dict.values())
        self.create_combined_df(attack_type, attack_stores, run_number)
        self.create_combined_best_run_fpr0001(attack_type, attack_stores,
                                              run_number)
        self.create_combined_best_run_fpr01(attack_type, attack_stores,
                                            run_number)
        self.create_combined_best_run_auc(attack_type, attack_stores,
                                          run_number)

        self.create_combined_averaged_roc_curve(attack_type, attack_stores,
                                                run_number)

    def create_combined_best_run_auc(
        self,
        attack_type: AttackType,
        attack_stores: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        ds_name_list = []

        for store in attack_stores:
            ds_name_list.append(store.ds_name)

            best_attack = store.get_best_auc_run()
            fpr = best_attack.roc_curve.fpr
            tpr = best_attack.roc_curve.tpr
            auc = best_attack.get_auc()
            ax.plot(fpr, tpr, label=f"{store.ds_name} AUC={auc:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text("Receiver Operator Characteristics - Best Run AUC")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_best_run_auc_{'-'.join(ds_name_list)}_results.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined best attack run AUC ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr0001(
        self,
        attack_type: AttackType,
        attack_stores: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        ds_name_list = []

        for store in attack_stores:
            ds_name_list.append(store.ds_name)

            best_attack = store.get_best_fpr001_run()
            _, fpr_0001 = store.get_fpr_at_fixed_tpr(best_attack)
            fpr = best_attack.roc_curve.fpr
            tpr = best_attack.roc_curve.tpr
            ax.plot(fpr,
                    tpr,
                    label=f"{store.ds_name} FPR@0.001={fpr_0001:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text(
            "Receiver Operator Characteristics - Best Run FPR@0.001")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_best_run_fpr0001_{'-'.join(ds_name_list)}_results.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined best attack run fpr0001 ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr01(
        self,
        attack_type: AttackType,
        attack_stores: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        ds_name_list = []

        for store in attack_stores:
            ds_name_list.append(store.ds_name)

            best_attack = store.get_best_fpr01_run()
            fpr_01, _ = store.get_fpr_at_fixed_tpr(best_attack)
            fpr = best_attack.roc_curve.fpr
            tpr = best_attack.roc_curve.tpr
            ax.plot(fpr, tpr, label=f"{store.ds_name} FPR@0.1={fpr_01:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text(
            "Receiver Operator Characteristics - Best Run FPR@0.1")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_best_run_fpr01_{'-'.join(ds_name_list)}_results.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined best attack run fpr01 ROC curve {plt_name}")
        plt.close()

    def create_combined_df(
        self,
        attack_type: AttackType,
        attack_stores: List[AttackResultStore],
        run_number: int,
    ):
        combined_df = pd.DataFrame()

        ds_names = []
        for store in attack_stores:
            ds_attack_result_df = store.attack_result_df[
                store.attack_result_df["shadow model"].isnull()]
            ds_attack_result_df["dataset"] = store.ds_name
            ds_attack_result_df = ds_attack_result_df[[
                "dataset",
                "attack type",
                "Attacker advantage",
                "Positive predictive value",
                "AUC",
                "fpr@0.1",
                "fpr@0.001",
            ]]
            combined_df = pd.concat([combined_df, ds_attack_result_df])
            ds_names.append(store.ds_name)

        file_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"combined_df_{'_'.join(ds_names)}.csv",
        )
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save_dataframe(combined_df.round(decimals=5), filename=file_name)

    def create_combined_averaged_roc_curve(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        name_list = []

        for store in attack_store:
            name_list.append(store.ds_name)
            avg_auc = store.attack_result_df.loc["mean Entire dataset"]["AUC"]
            avg_fpr01 = store.attack_result_df.loc["mean Entire dataset"][
                "fpr@0.1"]
            avg_fpr0001 = store.attack_result_df.loc["mean Entire dataset"][
                "fpr@0.001"]
            ax.plot(
                store.fpr_grid,
                store.mean_tpr,
                label=
                f"{store.ds_name} AUC={avg_auc:.3f} FPR@0.1={avg_fpr01:.3f} FPR@0.001={avg_fpr0001:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text("Receiver Operator Characteristics - Averaged")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_average_{'-'.join(name_list)}_results.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined averaged DS ROC curve {plt_name}")
        plt.close()

    def create_runs_combined_graphics(
        self,
        attack_type: AttackType,
        run_numbers: List[int],
        run_ds_name_dict: Dict[int, List[str]],
    ):
        """Create a comparison of averaged ROC curves for Entire Dataset attacks for every dataset.

        So this function creates a compilation of attack results over multiple runs.
        """
        avg_run_dict: Dict[str, List[AttackResultStore]] = defaultdict(list)

        for run in run_numbers:
            ds_list = run_ds_name_dict[run]

            result_dict = self.load_attack_results(attack_type,
                                                   run,
                                                   ds_name_list=ds_list)

            for ds_name, store in result_dict.items():
                # remove modifications from ds_name
                if "_" in ds_name:
                    ds_name = ds_name.split("_")[0]

                avg_run_dict[ds_name].append(store)

        # create an average ROC curve, where all datasets are averaged for a specific run
        # this ROC curve shall compare the average of all datasets between the runs
        self.create_compiled_averaged_run_roc_curves(
            attack_type=attack_type,
            avg_run_dict=avg_run_dict,
            runs=run_numbers,
        )

        for ds_name, store_list in avg_run_dict.items():
            self.create_combined_averaged_roc_curve_from_list(attack_type,
                                                              store_list,
                                                              runs=run_numbers,
                                                              ds_name=ds_name)
            #self.create_combined_average_class_rocs(attack_type,
            #                                        store_list,
            #                                        runs=run_numbers,
            #                                        ds_name=ds_name)

    def create_compiled_averaged_run_roc_curves(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """This function creates a ROC curve for every run representing the average of all datasets per run."""

        # a dict holding the list of attackresult stores for all datasets of this specific run
        run_dict: Dict[int, List[AttackResultStore]] = defaultdict(list)

        for k, attack_store_list in avg_run_dict.items():
            for i in range(len(attack_store_list)):
                run_dict[i].append(attack_store_list[i])

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        main_fpr_grid = None
        for k, attack_store_list in run_dict.items():
            mean_tpr_list: List[np.ndarray] = []

            mean_auc_list: List[float] = []
            mean_fpr01_list: List[float] = []
            mean_fpr0001_list: List[float] = []

            for store in attack_store_list:
                mean_auc_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]["AUC"])
                mean_fpr01_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]
                    ["fpr@0.1"])
                mean_fpr0001_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]
                    ["fpr@0.001"])

                entire_dataset_result_list = store.get_single_entire_ds_attack_results(
                )

                if main_fpr_grid is None:
                    tpr_mean, _, main_fpr_grid = store.calculate_mean_tpr_and_fpr(
                        entire_dataset_result_list)
                else:
                    tpr_mean, _, _ = store.calculate_mean_tpr_and_fpr(
                        entire_dataset_result_list, main_fpr_grid)

                mean_tpr_list.append(tpr_mean)

            # convert list to numpy array
            np_global_mean_tpr = np.vstack(mean_tpr_list)
            np_global_mean_tpr = np.mean(mean_tpr_list, axis=0)

            avg_auc = sum(mean_auc_list) / len(mean_auc_list)
            avg_fpr01 = sum(mean_fpr01_list) / len(mean_fpr01_list)
            avg_fpr0001 = sum(mean_fpr0001_list) / len(mean_fpr0001_list)

            ax.plot(
                main_fpr_grid,
                np_global_mean_tpr,
                label=
                f"Run {runs[k]} - AUC={avg_auc:.3f} FPR@0.1={avg_fpr01:.3f} FPR@0.001={avg_fpr0001:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text("Receiver Operator Characteristics - Run Averaged")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"roc_run_averaged_all_datasets_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved test ROC curve {plt_name}")
        plt.close()

    def create_combined_average_class_rocs(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        runs: List[int],
        ds_name: Optional[str] = None,
    ):
        run_dict = {}

        for store in attack_store:
            class_attack_dict = store.get_single_class_attack_result_dict()

            class_wise_mean_tpr_dict: Dict[str, Tuple[np.ndarray,
                                                      np.ndarray]] = {}

            for class_number, single_result_list in class_attack_dict.items():
                tpr_mean, _, fpr_grid = store.calculate_mean_tpr_and_fpr(
                    single_result_list)
                class_wise_mean_tpr_dict[class_number] = (tpr_mean, fpr_grid)

            run_dict[store.run_number] = class_wise_mean_tpr_dict

        if ds_name is None:
            ds_name = attack_store[0].ds_name

        fig, axs = plt.subplots(len(runs), figsize=(7, 5 * len(runs)))
        fig.suptitle(
            f"Receiver Operator Characteristics - Class-wise Attack ({attack_type.value}, {ds_name})"
        )

        for run_number, class_wise_dict in run_dict.items():
            axs[run_number].plot([0, 1], [0, 1], "k--", lw=1.0)
            for class_number, mean_values in class_wise_dict.items():
                axs[run_number].plot(mean_values[1],
                                     mean_values[0],
                                     label=f"Class {class_number}")
                axs[run_number].set(xlabel="FPR", ylabel="TPR")
                axs[run_number].set(aspect=1, xscale="log", yscale="log")
                axs[run_number].title.set_text(f"Run {run_number}")
                axs[run_number].legend(loc="lower right")
                axs[run_number].set_xlim([0.00001, 1])
                axs[run_number].set_ylim([0.00001, 1])

        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"roc_combined_average_{ds_name}_results_class_wise_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined averaged DS ROC curve {plt_name} (class wise)")
        plt.close()

    def create_combined_averaged_roc_curve_from_list(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        runs: List[int],
        ds_name: Optional[str] = None,
    ):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        for store in attack_store:
            entire_dataset_result_list = store.get_single_entire_ds_attack_results(
            )
            tpr_mean, _, fpr_grid = store.calculate_mean_tpr_and_fpr(
                entire_dataset_result_list)

            avg_auc = store.attack_result_df.loc["mean Entire dataset"]["AUC"]
            fpr0001 = store.attack_result_df.loc["mean Entire dataset"][
                "fpr@0.001"]
            fpr01 = store.attack_result_df.loc["mean Entire dataset"][
                "fpr@0.1"]
            ax.plot(
                fpr_grid,
                tpr_mean,
                label=
                f"Run {store.run_number} AUC={avg_auc:.3f} FPR@0.1={fpr01:.3f} FPR@0.001={fpr0001:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")

        if ds_name is None:
            ds_name = attack_store[0].ds_name

        ax.title.set_text(
            f"Receiver Operator Characteristics - All Runs ({attack_type.value}, {ds_name})"
        )
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"roc_combined_average_{ds_name}_results_entire_dataset_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined averaged DS ROC curve {plt_name}")
        plt.close()
