import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attack_result_store import AttackResultStore, AttackType
from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import save_dataframe

pd.options.mode.chained_assignment = None

FIGSIZE = (5, 3)

AXHLINE_COLOR = "tab:gray"
AXHLINE_WIDTH = 2.0
AXHLINE_STYLE = "-"
LEGEND_ALPHA = 0.50


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

        self.x_axis_name = settings.x_axis_name
        self.x_axis_values = settings.x_axis_values

        # base folder where all individual runs are saved
        self.run_result_folder = os.path.join(
            result_path, settings.model_name, settings.run_name
        )

        self.analysis_combined_runs = os.path.join(
            result_path,
            settings.model_name,
            settings.run_name,
            "attack-analysis-combined",
        )
        check_create_folder(self.analysis_combined_runs)

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
            print("No analysis run number range specified! Aborting...")
            sys.exit(1)

        runs = self.settings.analysis_run_numbers

        run_ds_name_dict: Dict[int, List[str]] = {}

        run_combined_df = None
        for run in runs:
            print(
                f" --- Creating attack analysis for run: {self.settings.run_name} -> {run} ---"
            )
            # load pararmeter.json and ds_name_list
            parameter_file_path = os.path.join(
                self.run_result_folder, str(run), "parameter_model_attack.json"
            )
            ds_list: List[str] = None
            try:
                with open(parameter_file_path) as parameter_file:
                    parameter_dict = json.load(parameter_file)
                    ds_list = parameter_dict["datasets"]
                    run_ds_name_dict[run] = ds_list
            except FileNotFoundError:
                parameter_file_path_2 = os.path.join(
                    self.run_result_folder, str(run), "parameter.json"
                )
                print(
                    f"Could not find file: {parameter_file_path}. Trying other file: {parameter_file_path_2}"
                )
                with open(parameter_file_path_2) as parameter_file:
                    parameter_dict = json.load(parameter_file)
                    ds_list = parameter_dict["datasets"]
                    run_ds_name_dict[run] = ds_list

            if ds_list is None:
                print(
                    "Cannot load attack results, since the parameter_model_attack.json/ parameter.json file does not contain used dataset names for the runs!"
                )
                sys.exit(1)

            result_dict = self.load_attack_results(
                attack_type=attack_type, run_number=run, ds_name_list=ds_list
            )
            df = self._compile_attack_results(attack_type, result_dict, run)
            df = df.loc[["mean Entire dataset", "average"]]
            if run_combined_df is None:
                run_combined_df = df
            else:
                run_combined_df = pd.concat([run_combined_df, df])

        run_combined_df_filepath = os.path.join(
            self.analysis_combined_runs, "attack_metrics_combined.csv"
        )
        save_dataframe(run_combined_df, filename=run_combined_df_filepath)

        # only combine multiple runs if there are any
        if len(runs) > 1:
            print(" --- Creating combined runs graphics ---")
            self.create_runs_combined_graphics(attack_type, runs, run_ds_name_dict)

    def _compile_attack_results(
        self,
        attack_type: AttackType,
        attack_result_dict: Dict[str, AttackResultStore],
        run_number: int,
    ) -> pd.DataFrame:
        """Create an overview of compiled attack results for a single run."""
        for ds_name, store in attack_result_dict.items():
            # store.create_entire_dataset_combined_roc_curve()

            # this function sets the store's mean_tpr and fpr_grid
            store.create_average_roc_curve_entire_dataset(generate_std_area=True)
            store.create_average_class_attack_roc()

        # create figures to compare the best runs of each dataset with each other
        attack_stores = list(attack_result_dict.values())
        df = self.create_combined_df(attack_type, attack_stores, run_number)

        # dont currently create the best run graphics
        # self.create_combined_best_run_fpr0001(attack_type, attack_stores,
        #                                      run_number)
        # self.create_combined_best_run_fpr01(attack_type, attack_stores,
        #                                    run_number)
        # self.create_combined_best_run_auc(attack_type, attack_stores,
        #                                  run_number)

        self.create_combined_averaged_roc_curve(attack_type, attack_stores, run_number)
        self.create_combined_attack_metric_bar_chart(
            attack_type, attack_stores, run_number
        )

        return df

    def create_combined_best_run_auc(
        self,
        attack_type: AttackType,
        attack_stores: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
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
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend(framealpha=LEGEND_ALPHA)

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_best_run_auc_{'-'.join(ds_name_list)}_results_{attack_type.value}.png",
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
        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        ds_name_list = []

        for store in attack_stores:
            ds_name_list.append(store.ds_name)

            best_attack = store.get_best_fpr001_run()
            _, fpr_0001 = store.get_fpr_at_fixed_tpr(best_attack)
            fpr = best_attack.roc_curve.fpr
            tpr = best_attack.roc_curve.tpr
            ax.plot(fpr, tpr, label=f"{store.ds_name} TPR@0.001={fpr_0001:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend(framealpha=LEGEND_ALPHA)

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
        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        ds_name_list = []

        for store in attack_stores:
            ds_name_list.append(store.ds_name)

            best_attack = store.get_best_fpr01_run()
            fpr_01, _ = store.get_fpr_at_fixed_tpr(best_attack)
            fpr = best_attack.roc_curve.fpr
            tpr = best_attack.roc_curve.tpr
            ax.plot(fpr, tpr, label=f"{store.ds_name} TPR@0.1={fpr_01:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend(framealpha=LEGEND_ALPHA)

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
    ) -> pd.DataFrame:
        combined_df = pd.DataFrame()

        ds_names = []
        for store in attack_stores:
            ds_attack_result_df = store.attack_result_df[
                store.attack_result_df["shadow model"].isnull()
            ]
            ds_attack_result_df["dataset"] = store.ds_name
            ds_attack_result_df = ds_attack_result_df[
                [
                    "dataset",
                    "attack type",
                    # "Attacker advantage",
                    # "Positive predictive value",
                    "AUC",
                    "fpr@0.1",
                    "fpr@0.001",
                ]
            ]
            combined_df = pd.concat([combined_df, ds_attack_result_df])
            ds_names.append(store.ds_name)

        # create average attack values
        avg_auc = combined_df.loc["mean Entire dataset"]["AUC"].mean(axis=0)
        avg_fpr01 = combined_df.loc["mean Entire dataset"]["fpr@0.1"].mean(axis=0)
        avg_fpr0001 = combined_df.loc["mean Entire dataset"]["fpr@0.001"].mean(axis=0)

        average_row = ["average", "", avg_auc, avg_fpr01, avg_fpr0001]
        combined_df.loc["average"] = average_row

        file_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"combined_df_{'_'.join(ds_names)}_{attack_type.value}.csv",
        )
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save_dataframe(combined_df.round(decimals=4), filename=file_name)

        return combined_df

    def create_combined_attack_metric_bar_chart(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        run_number: int,
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

        class_wise_auc_dict: Dict[str, List[float]] = defaultdict(list)
        class_wise_fpr01_dict: Dict[str, List[float]] = defaultdict(list)
        class_wise_fpr0001_dict: Dict[str, List[float]] = defaultdict(list)

        for store in attack_store:
            name = store.ds_name
            if "" in name:
                if "_gray" in name:
                    name = name.split("_")[0] + "_gray"
                elif "_" in name:
                    name = name.split("_")[0]

            # get number of classes:
            num_classes = store.attack_result_df[
                store.attack_result_df["slice value"].fillna(value=" ").str.isnumeric()
            ]["slice value"].max()
            num_classes = int(num_classes) + 1

            for class_i in range(num_classes):
                class_values = store.attack_result_df.loc[f"mean CLASS={class_i}"]
                class_auc = class_values["AUC"]
                class_fpr01 = class_values["fpr@0.1"]
                class_fpr0001 = class_values["fpr@0.001"]

                class_wise_auc_dict[name].append(class_auc)
                class_wise_fpr01_dict[name].append(class_fpr01)
                class_wise_fpr0001_dict[name].append(class_fpr0001)

        fig_auc, ax_auc = plt.subplots(figsize=FIGSIZE, layout="constrained")
        fig_fpr01, ax_fpr01 = plt.subplots(figsize=FIGSIZE, layout="constrained")
        fig_fpr0001, ax_fpr0001 = plt.subplots(figsize=FIGSIZE, layout="constrained")

        x = np.arange(len(list(class_wise_auc_dict.values())[0]))
        width = 0.22
        multiplier = 0

        name_list = []
        for ds_name, values in class_wise_auc_dict.items():
            name_list.append(ds_name)

            offset = width * multiplier
            ax_auc.bar(x + offset, values, width, label=ds_name)
            multiplier += 1

        multiplier = 0
        for ds_name, values in class_wise_fpr01_dict.items():
            offset = width * multiplier
            ax_fpr01.bar(x + offset, values, width, label=ds_name)
            multiplier += 1

        multiplier = 0
        for ds_name, values in class_wise_fpr0001_dict.items():
            offset = width * multiplier
            ax_fpr0001.bar(x + offset, values, width, label=ds_name)
            multiplier += 1

        ax_auc.set_ylabel("AUC")
        ax_auc.set_xlabel("Classes")
        ax_auc.set_xticks(x + width, x)
        ax_auc.legend(loc="lower right", framealpha=LEGEND_ALPHA)

        auc_bar_chart_fn: str = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"bar_chart_class_wise_AUC_{''.join(name_list)}_r{run_number}.png",
        )
        print(f"Saving class wise bar chart AUC figure to {auc_bar_chart_fn}")
        fig_auc.savefig(auc_bar_chart_fn)

        ax_fpr01.set_ylabel("TPR@0.1")
        ax_fpr01.set_xlabel("Classes")
        ax_fpr01.set_xticks(x + width, x)
        ax_fpr01.legend(framealpha=LEGEND_ALPHA)

        fpr01_bar_chart_fn: str = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"bar_chart_class_wise_FPR01_{''.join(name_list)}_r{run_number}.png",
        )
        print(f"Saving class wise bar chart TPR@0.1 figure to {fpr01_bar_chart_fn}")
        fig_fpr01.savefig(fpr01_bar_chart_fn)

        ax_fpr0001.set_ylabel("TPR@0.001")
        ax_fpr0001.set_xlabel("Classes")
        ax_fpr0001.set_xticks(x + width, x)
        ax_fpr0001.legend(framealpha=LEGEND_ALPHA)

        fpr0001_bar_chart_fn: str = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"bar_chart_class_wise_FPR0001_{''.join(name_list)}_r{run_number}.png",
        )
        print(f"Saving class wise bar chart TPR@0.001 figure to {fpr0001_bar_chart_fn}")
        fig_fpr0001.savefig(fpr0001_bar_chart_fn)

        plt.close()

    def create_combined_averaged_roc_curve(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        run_number: int,
    ):
        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        name_list = []

        for store in attack_store:
            name = store.ds_name
            if "" in name:
                # name = name.split("_")[0]

                if "_gray" in name:
                    name = name.split("_")[0] + "_gray"
                elif "_" in name:
                    name = name.split("_")[0]

            name_list.append(name)
            avg_auc = store.attack_result_df.loc["mean Entire dataset"]["AUC"]

            ax.plot(
                store.fpr_grid,
                store.mean_tpr,
                label=f"{name} AUC={avg_auc:.3f}",
                # f"{store.ds_name} AUC={avg_auc:.3f} FPR@0.1={avg_fpr01:.3f} FPR@0.001={avg_fpr0001:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])
        plt.legend(framealpha=LEGEND_ALPHA)

        plt_name = os.path.join(
            self.get_combined_ds_analysis_folder(run_number),
            f"roc_combined_average_{'-'.join(name_list)}_results_{attack_type.value}.png",
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

            result_dict = self.load_attack_results(
                attack_type, run, ds_name_list=ds_list
            )

            for ds_name, store in result_dict.items():
                # remove modifications from ds_name
                if "_gray" in ds_name:
                    ds_name = ds_name.split("_")[0] + "_gray"
                elif "_" in ds_name:
                    ds_name = ds_name.split("_")[0]

                avg_run_dict[ds_name].append(store)

        self.create_combined_average_class_rocs(
            attack_type, avg_run_dict, runs=run_numbers, ds_name=ds_name
        )

        self.create_compiled_auc_metric_graph(
            attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        )
        # self.create_compiled_max_auc_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )
        # self.create_compiled_std_auc_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )

        self.create_compiled_fpr01_metric_graph(
            attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        )
        # self.create_compiled_max_fpr01_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )
        # self.create_compiled_std_fpr01_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )

        self.create_compiled_fpr0001_metric_graph(
            attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        )
        # self.create_compiled_max_fpr0001_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )
        # self.create_compiled_std_fpr0001_metric_graph(
        #    attack_type=attack_type, avg_run_dict=avg_run_dict, runs=run_numbers
        # )

        for ds_name, store_list in avg_run_dict.items():
            self.create_combined_averaged_roc_curve_from_list(
                attack_type, store_list, runs=run_numbers, ds_name=ds_name
            )

        # create an average ROC curve, where all datasets are averaged for a specific run
        # this ROC curve shall compare the average of all datasets between the runs
        self.create_compiled_averaged_run_roc_curves(
            attack_type=attack_type,
            avg_run_dict=avg_run_dict,
            runs=run_numbers,
        )

    def create_combined_average_class_rocs(
        self,
        attack_type: AttackType,
        attack_dict: Dict[str, List[AttackResultStore]],
        runs: List[int],
        ds_name: Optional[str] = None,
    ):
        """Create an anverage ROC curve for every class from every dataset and for every run."""

        # a dict containing the attack results
        # the run number is used as dict key
        run_dict: Dict[int, List[AttackResultStore]] = defaultdict(list)

        for ds_name, attack_result_list in attack_dict.items():
            for i, attack_results in enumerate(attack_result_list):
                run_dict[i].append(attack_results)

        # calc average values from auc, fpr, ...
        # a dict containing for every class the average metric over runs
        avg_auc_dict: Dict[str, List[float]] = defaultdict(list)
        avg_fpr01_dict: Dict[str, List[float]] = defaultdict(list)
        avg_fpr0001_dict: Dict[str, List[float]] = defaultdict(list)

        # create average per run
        for run, attack_store_list in run_dict.items():
            fpr_grid = None
            # a dict containing the average TPR for each class
            class_wise_mean_tpr_dict: Dict[str, List[np.ndarray]] = defaultdict(list)

            class_auc_dict: Dict[str, List[float]] = defaultdict(list)
            class_fpr01_dict: Dict[str, List[float]] = defaultdict(list)
            class_fpr0001_dict: Dict[str, List[float]] = defaultdict(list)

            for store in attack_store_list:
                class_attack_dict = store.get_single_class_attack_result_dict()
                for class_number, single_result_list in class_attack_dict.items():
                    class_auc_dict[class_number].append(
                        store.attack_result_df.loc[f"mean CLASS={class_number}"]["AUC"]
                    )
                    class_fpr01_dict[class_number].append(
                        store.attack_result_df.loc[f"mean CLASS={class_number}"][
                            "fpr@0.1"
                        ]
                    )
                    class_fpr0001_dict[class_number].append(
                        store.attack_result_df.loc[f"mean CLASS={class_number}"][
                            "fpr@0.001"
                        ]
                    )

                    if fpr_grid is None:
                        tpr_mean, _, fpr_grid = store.calculate_mean_tpr_and_fpr(
                            single_result_list
                        )
                    else:
                        tpr_mean, _, _ = store.calculate_mean_tpr_and_fpr(
                            single_result_list, fpr_grid=fpr_grid
                        )

                    class_wise_mean_tpr_dict[class_number].append(tpr_mean)

            for k, v in class_auc_dict.items():
                avg_auc_dict[k].append(np.mean(v))

            for k, v in class_fpr01_dict.items():
                avg_fpr01_dict[k].append(np.mean(v))

            for k, v in class_fpr0001_dict.items():
                avg_fpr0001_dict[k].append(np.mean(v))

            mean_class_tpr: Dict[str, np.ndarray] = {}
            # create average from all values within a class from different datasets
            for class_number, class_tpr in class_wise_mean_tpr_dict.items():
                avg_array = np.stack(class_tpr, axis=0)
                avg_array = np.mean(avg_array, axis=0)
                mean_class_tpr[class_number] = avg_array

            fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
            ax.plot([0, 1], [0, 1], "k--", lw=1.0)

            for class_number, avg_tpr in mean_class_tpr.items():
                ax.plot(
                    fpr_grid,
                    avg_tpr,
                    label=f"Class {class_number} AUC={avg_auc_dict[class_number][-1]:.3f}",
                )
                ax.set(xlabel="FPR", ylabel="TPR")
                ax.set(aspect=1, xscale="log", yscale="log")
                ax.legend(framealpha=LEGEND_ALPHA)
                ax.set_xlim([0.0001, 1])
                ax.set_ylim([0.0001, 1])

            plt_name = os.path.join(
                self.analysis_combined_runs,
                f"roc_average_classes_{attack_type.value}_run_{run}.png",
            )
            os.makedirs(os.path.dirname(plt_name), exist_ok=True)
            plt.savefig(plt_name)
            print(f"Saved combined averaged DS ROC curve {plt_name} (class wise)")

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        fig_auc, ax_auc = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for k, v in avg_auc_dict.items():
            # handle the case for the class count experiment, pad v to math x_values length
            if len(x_values) != len(v):
                v += [0] * (len(x_values) - len(v))

            ax_auc.plot(
                x_values,
                v,
                label=f"Class {k}",
                marker="x",
            )
        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_auc.set(xlabel=x_name, ylabel="AUC")
        plt.xticks(x_values)
        plt.legend(
            loc="lower left",
            labelspacing=0.4,
            columnspacing=1,
            framealpha=LEGEND_ALPHA,
            handlelength=1.2,
            handletextpad=0.3,
            ncols=3,
            fontsize="small",
            markerscale=0.8,
        )
        ax_auc.axhline(
            y=0.5, linestyle=AXHLINE_STYLE, color=AXHLINE_COLOR, linewidth=AXHLINE_WIDTH
        )
        plt.grid(True)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"class_wise_auc_over_runs_{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled class wise AUC over all runs graph {plt_name}")

        fig_fpr01, ax_fpr01 = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for k, v in avg_fpr01_dict.items():
            # handle the case for the class count experiment, pad v to math x_values length
            if len(x_values) != len(v):
                v += [0] * (len(x_values) - len(v))
            ax_fpr01.plot(
                x_values,
                v,
                label=f"Class {k}",
                marker="x",
            )
        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr01.set(xlabel=x_name, ylabel="TPR@0.1")
        plt.xticks(x_values)
        plt.legend(
            loc="lower left",
            labelspacing=0.4,
            columnspacing=1,
            framealpha=LEGEND_ALPHA,
            handlelength=1.2,
            handletextpad=0.3,
            ncols=3,
            fontsize="small",
            markerscale=0.8,
        )
        ax_fpr01.axhline(
            y=0.1, linestyle=AXHLINE_STYLE, color=AXHLINE_COLOR, linewidth=AXHLINE_WIDTH
        )
        plt.grid(True)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"class_wise_fpr01_over_runs_{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled class wise TPR@0.1 over all runs graph {plt_name}")

        fig_fpr001, ax_fpr0001 = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for k, v in avg_fpr0001_dict.items():
            # handle the case for the class count experiment, pad v to math x_values length
            if len(x_values) != len(v):
                v += [0] * (len(x_values) - len(v))
            ax_fpr0001.plot(
                x_values,
                v,
                label=f"Class {k}",
                marker="x",
            )
        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr0001.set(xlabel=x_name, ylabel="TPR@0.001")
        plt.xticks(x_values)
        plt.legend(
            loc="lower left",
            labelspacing=0.4,
            columnspacing=1,
            framealpha=LEGEND_ALPHA,
            handlelength=1.2,
            handletextpad=0.3,
            ncols=3,
            fontsize="small",
            markerscale=0.8,
        )
        plt.grid(True)
        ax_fpr0001.axhline(
            y=0.001,
            linestyle=AXHLINE_STYLE,
            color=AXHLINE_COLOR,
            linewidth=AXHLINE_WIDTH,
        )
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"class_wise_fpr0001_over_runs_{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled class wise TPR@0.001 over all runs graph {plt_name}")

        plt.close()

    def create_compiled_auc_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with AUC for every dataset in every run."""
        _, ax_auc = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            auc_list: List[float] = []

            for store in attack_store_list:
                auc_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]["AUC"]
                )

            mean_list.append(auc_list)
            ax_auc.plot(
                x_values,
                auc_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)

        ax_auc.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )
        ax_auc.axhline(
            y=0.5, linestyle=AXHLINE_STYLE, color=AXHLINE_COLOR, linewidth=AXHLINE_WIDTH
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name

        ax_auc.set(xlabel=x_name, ylabel="AUC")
        plt.xticks(x_values)
        # plt.yticks(np.arange(0.50, 0.9, 0.05))
        plt.legend(framealpha=LEGEND_ALPHA)
        plt.grid(True)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"avg_auc_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled AUC over all runs graph {plt_name}")
        plt.close()

    def create_compiled_fpr01_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.01 for every dataset in every run."""
        _, ax_fpr01 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr01_list: List[float] = []

            for store in attack_store_list:
                fpr01_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]["fpr@0.1"]
                )

            mean_list.append(fpr01_list)
            ax_fpr01.plot(
                x_values,
                fpr01_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr01.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )
        ax_fpr01.axhline(
            y=0.1, linestyle=AXHLINE_STYLE, color=AXHLINE_COLOR, linewidth=AXHLINE_WIDTH
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr01.set(xlabel=x_name, ylabel="TPR@0.1")
        plt.xticks(x_values)
        # plt.yticks(np.arange(0.1, 0.6, 0.05))
        plt.grid(True)
        ax_fpr01.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"avg_fpr01_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.1 over all runs graph {plt_name}")
        plt.close()

    def create_compiled_max_fpr01_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.01 for every dataset in every run."""
        _, ax_fpr01 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr01_list: List[float] = []

            for store in attack_store_list:
                fpr01_list.append(
                    store.attack_result_df.loc["max Entire dataset"]["fpr@0.1"]
                )

            mean_list.append(fpr01_list)
            ax_fpr01.plot(
                x_values,
                fpr01_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr01.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr01.set(xlabel=x_name, ylabel="TPR@0.1")
        plt.xticks(x_values)
        plt.grid(True)
        ax_fpr01.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"max_fpr01_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.1 max over all runs graph {plt_name}")
        plt.close()

    def create_compiled_std_fpr01_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.01 for every dataset in every run."""
        _, ax_fpr01 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr01_list: List[float] = []

            for store in attack_store_list:
                fpr01_list.append(
                    store.attack_result_df.loc["std Entire dataset"]["fpr@0.1"]
                )

            mean_list.append(fpr01_list)
            ax_fpr01.plot(
                x_values,
                fpr01_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr01.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr01.set(xlabel=x_name, ylabel="TPR@0.1")
        plt.xticks(x_values)
        plt.grid(True)
        ax_fpr01.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"std_fpr01_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.1 std over all runs graph {plt_name}")
        plt.close()

    def create_compiled_fpr0001_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.001 for every dataset in every run."""
        _, ax_fpr0001 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr0001_list: List[float] = []

            for store in attack_store_list:
                fpr0001_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]["fpr@0.001"]
                )

            mean_list.append(fpr0001_list)

            ax_fpr0001.plot(
                x_values,
                fpr0001_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr0001.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        ax_fpr0001.axhline(
            y=0.001,
            linestyle=AXHLINE_STYLE,
            color=AXHLINE_COLOR,
            linewidth=AXHLINE_WIDTH,
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr0001.set(xlabel=x_name, ylabel="TPR@0.001")
        plt.xticks(x_values)
        # plt.yticks(np.arange(0.00, 0.15, 0.02))
        plt.grid(True)
        plt.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"avg_fpr0001_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.001 over all runs graph {plt_name}")
        plt.close()

    def create_compiled_max_fpr0001_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.001 for every dataset in every run."""
        _, ax_fpr0001 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr0001_list: List[float] = []

            for store in attack_store_list:
                fpr0001_list.append(
                    store.attack_result_df.loc["max Entire dataset"]["fpr@0.001"]
                )

            mean_list.append(fpr0001_list)

            ax_fpr0001.plot(
                x_values,
                fpr0001_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr0001.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr0001.set(xlabel=x_name, ylabel="TPR@0.001")
        plt.xticks(x_values)
        plt.grid(True)
        plt.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"max_fpr0001_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.001 max over all runs graph {plt_name}")
        plt.close()

    def create_compiled_std_fpr0001_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with fpr@0.001 for every dataset in every run."""
        _, ax_fpr0001 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            fpr0001_list: List[float] = []

            for store in attack_store_list:
                fpr0001_list.append(
                    store.attack_result_df.loc["std Entire dataset"]["fpr@0.001"]
                )

            mean_list.append(fpr0001_list)

            ax_fpr0001.plot(
                x_values,
                fpr0001_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)
        ax_fpr0001.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name
        ax_fpr0001.set(xlabel=x_name, ylabel="TPR@0.001")
        plt.xticks(x_values)
        plt.grid(True)
        plt.legend(framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"std_fpr0001_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled fpr@0.001 std over all runs graph {plt_name}")
        plt.close()

    def create_compiled_averaged_run_roc_curves(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """This function creates a ROC curve for every run representing the average of all datasets per run."""

        # a dict holding the list of attackresult stores for all datasets of this specific run
        run_dict: Dict[int, List[AttackResultStore]] = defaultdict(list)

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            for i in self.settings.analysis_run_numbers:
                run_dict[i].append(attack_store_list[i])

        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        main_fpr_grid = None
        for k, attack_store_list in run_dict.items():
            mean_tpr_list: List[np.ndarray] = []

            mean_auc_list: List[float] = []

            for store in attack_store_list:
                mean_auc_list.append(
                    store.attack_result_df.loc["mean Entire dataset"]["AUC"]
                )

                entire_dataset_result_list = store.get_single_entire_ds_attack_results()

                if main_fpr_grid is None:
                    tpr_mean, _, main_fpr_grid = store.calculate_mean_tpr_and_fpr(
                        entire_dataset_result_list
                    )
                else:
                    tpr_mean, _, _ = store.calculate_mean_tpr_and_fpr(
                        entire_dataset_result_list, main_fpr_grid
                    )

                mean_tpr_list.append(tpr_mean)

            # convert list to numpy array
            np_global_mean_tpr = np.vstack(mean_tpr_list)
            np_global_mean_tpr = np.mean(mean_tpr_list, axis=0)

            avg_auc = sum(mean_auc_list) / len(mean_auc_list)

            ax.plot(
                main_fpr_grid,
                np_global_mean_tpr,
                label=f"{x_values[k]} AUC={avg_auc:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])

        title = None
        if self.x_axis_name is not None:
            title = self.x_axis_name

        plt.legend(title=title, framealpha=LEGEND_ALPHA)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"roc_run_averaged_all_datasets_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved all DS averaged ROC curve {plt_name}")
        plt.close()

    def create_combined_averaged_roc_curve_from_list(
        self,
        attack_type: AttackType,
        attack_store: List[AttackResultStore],
        runs: List[int],
        ds_name: Optional[str] = None,
    ):
        _, ax = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for store in attack_store:
            entire_dataset_result_list = store.get_single_entire_ds_attack_results()
            tpr_mean, _, fpr_grid = store.calculate_mean_tpr_and_fpr(
                entire_dataset_result_list
            )

            avg_auc = store.attack_result_df.loc["mean Entire dataset"]["AUC"]
            ax.plot(
                fpr_grid,
                tpr_mean,
                label=f"{x_values[store.run_number]} AUC={avg_auc:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")

        if ds_name is None:
            ds_name = attack_store[0].ds_name

        plt.xlim([0.0001, 1])
        plt.ylim([0.0001, 1])

        title = None
        if self.x_axis_name is not None:
            title = self.x_axis_name

        plt.legend(title=title, framealpha=LEGEND_ALPHA)

        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"roc_combined_average_{ds_name}_results_entire_dataset_r{''.join(map(str,runs))}_{attack_type.value}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved combined averaged DS ROC curve {plt_name}")
        plt.close()

    def create_compiled_max_auc_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with AUC for every dataset in every run."""
        _, ax_auc = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            auc_list: List[float] = []

            for store in attack_store_list:
                auc_list.append(store.attack_result_df.loc["max Entire dataset"]["AUC"])

            mean_list.append(auc_list)
            ax_auc.plot(
                x_values,
                auc_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)

        ax_auc.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name

        ax_auc.set(xlabel=x_name, ylabel="AUC")
        plt.xticks(x_values)
        plt.legend(framealpha=LEGEND_ALPHA)
        plt.grid(True)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"max_auc_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled AUC max over all runs graph {plt_name}")
        plt.close()

    def create_compiled_std_auc_metric_graph(
        self,
        avg_run_dict: Dict[str, List[AttackResultStore]],
        attack_type: AttackType,
        runs: List[int],
    ):
        """Create graph with AUC for every dataset in every run."""
        _, ax_auc = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

        mean_list: List[List[float]] = []

        x_values = runs
        if self.x_axis_values is not None:
            x_values = self.x_axis_values

        for ds_name, attack_store_list in avg_run_dict.items():
            auc_list: List[float] = []

            for store in attack_store_list:
                auc_list.append(store.attack_result_df.loc["std Entire dataset"]["AUC"])

            mean_list.append(auc_list)
            ax_auc.plot(
                x_values,
                auc_list,
                label=f"{ds_name}",
                marker="x",
            )

        np_mean = np.vstack(mean_list)
        np_mean = np.mean(np_mean, axis=0)

        ax_auc.plot(
            x_values,
            np_mean,
            linestyle="dashed",
            linewidth=3,
            label="average",
        )

        x_name = "Run"
        if self.x_axis_name is not None:
            x_name = self.x_axis_name

        ax_auc.set(xlabel=x_name, ylabel="AUC")
        plt.xticks(x_values)
        plt.legend(framealpha=LEGEND_ALPHA)
        plt.grid(True)
        plt_name = os.path.join(
            self.analysis_combined_runs,
            f"std_auc_over_runs_r{''.join(map(str,runs))}.png",
        )
        os.makedirs(os.path.dirname(plt_name), exist_ok=True)
        plt.savefig(plt_name)
        print(f"Saved compiled AUC std over all runs graph {plt_name}")
        plt.close()
