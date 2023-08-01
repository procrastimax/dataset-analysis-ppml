import os
import sys
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
    AttackResults,
    SingleAttackResult,
)

from attack_result_store import AttackResultStore, AttackType
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import find_nearest, save_dataframe

pd.options.mode.chained_assignment = None


class AttackAnalyser:

    def __init__(
        self,
        ds_list: List[AbstractDataset],
        settings: RunSettings,
        result_path: str,
        model_path: str,
    ):
        self.result_path = result_path
        check_create_folder(self.result_path)

        self.ds_list = ds_list

        self.settings: RunSettings = settings
        self.model_path: str = model_path

        # here are all the results after attacking the models with LIRA or MIA
        self.attack_statistics_folder: str = os.path.join(
            result_path,
            settings.model_name,
            settings.run_name,
            str(settings.run_number),
            "attack-statistics",
        )
        check_create_folder(self.attack_statistics_folder)

        self.attack_statistics_folder_combined: str = os.path.join(
            self.attack_statistics_folder, "combined-ds")
        check_create_folder(self.attack_statistics_folder_combined)

    def load_attack_results(self, attack_type: AttackType,
                            settings: RunSettings,
                            model_path: str) -> Dict[str, AttackResultStore]:
        attack_result_dict: Dict[str, AttackResultStore] = {}
        for ds in self.ds_list:
            model_save_path: str = os.path.join(
                model_path,
                settings.model_name,
                settings.run_name,
                str(settings.run_number),
                ds.dataset_name,
            )
            shadow_model_save_path: str = os.path.join(
                model_path,
                settings.model_name,
                settings.run_name,
                str(settings.run_number),
                "shadow_models",
                ds.dataset_name,
            )
            numpy_path: str = os.path.join(shadow_model_save_path, "data")
            in_indices_filename = os.path.join(numpy_path, "in_indices.pckl")
            stat_filename = os.path.join(numpy_path, "model_stat.pckl")
            loss_filename = os.path.join(numpy_path, "model_loss.pckl")

            attack_list_filename = os.path.join(
                self.attack_statistics_folder,
                "pickles",
                f"{ds.dataset_name}_attack_{attack_type.value}_results.pckl",
            )

            lira_results_store = AttackResultStore(
                attack_type=attack_type,
                shadow_model_dir=shadow_model_save_path,
                model_save_path=model_save_path,
                ds_name=ds.dataset_name,
                numpy_path=numpy_path,
                in_indices_filename=in_indices_filename,
                stat_filename=stat_filename,
                loss_filename=loss_filename,
                attack_result_list_filename=attack_list_filename,
                attack_result_base_folder=self.attack_statistics_folder,
            )
            attack_result_dict[ds.dataset_name] = lira_results_store
        return attack_result_dict

    def compile_attack_results_lira(self):
        attack_type: AttackType = AttackType.LIRA
        lira_result_dict = self.load_attack_results(attack_type,
                                                    settings=self.settings,
                                                    model_path=self.model_path)

        for ds_name, store in lira_result_dict.items():
            store.load_saved_values()

            result_df = store.create_complete_dataframe(
                store.attack_result_list)

            lira_result_dict[ds_name].attack_result_df = result_df

            df_filename = os.path.join(
                store.result_path,
                f"{attack_type.value}_attack_statistic_results_{store.ds_name}.csv",
            )
            save_dataframe(store.attack_result_df, df_filename)

            store.create_entire_dataset_combined_roc_curve()
            store.create_average_roc_curve_entire_dataset(
                generate_std_area=True)
            store.create_average_class_attack_roc()

        # create figures to compare the best runs of each dataset with each other
        attack_stores = list(lira_result_dict.values())
        self.create_combined_best_run_fpr0001(attack_stores)
        self.create_combined_best_run_fpr01(attack_stores)
        self.create_combined_best_run_auc(attack_stores)

        self.create_combined_df(attack_stores)
        self.create_combined_averaged_roc_curve(attack_stores)

    def create_combined_best_run_auc(self,
                                     attack_stores: List[AttackResultStore]):
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
        ax.title.set_text("ROC Combined Best Run AUC")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.attack_statistics_folder_combined,
            f"roc_combined_best_run_auc_{'-'.join(ds_name_list)}_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved combined best attack run AUC ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr0001(
            self, attack_stores: List[AttackResultStore]):
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
        ax.title.set_text("ROC Combined Best Run FPR@0.001")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.attack_statistics_folder_combined,
            f"roc_combined_best_run_fpr0001_{'-'.join(ds_name_list)}_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved combined best attack run fpr0001 ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr01(self,
                                       attack_stores: List[AttackResultStore]):
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
        ax.title.set_text("ROC Combined Best Run FPR@0.1")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.attack_statistics_folder_combined,
            f"roc_combined_best_run_fpr01_{'-'.join(ds_name_list)}_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved combined best attack run fpr01 ROC curve {plt_name}")
        plt.close()

    def create_combined_df(self, attack_stores: List[AttackResultStore]):
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

        print("Combined DF:")
        print(combined_df)
        file_name = os.path.join(
            self.attack_statistics_folder_combined,
            f"combined_df_{'_'.join(ds_names)}.csv",
        )
        save_dataframe(combined_df, filename=file_name)

    def create_combined_averaged_roc_curve(
            self, attack_store: List[AttackResultStore]):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)

        name_list = []

        for store in attack_store:
            name_list.append(store.ds_name)
            avg_auc = store.attack_result_df.loc["mean Entire dataset"]["AUC"]
            ax.plot(
                store.fpr_grid,
                store.mean_tpr,
                label=f"{store.ds_name} AUC={avg_auc:.3f}",
            )

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale="log", yscale="log")
        ax.title.set_text("Receiver Operator Characteristics Averaged")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(
            self.attack_statistics_folder_combined,
            f"averaged_roc_curve_{'-'.join(name_list)}_results.png",
        )
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()


class UtilityAnalyser:

    def __init__(self, result_path: str, run_name: str, model_name: str):
        self.run_name = run_name
        self.model_name = model_name
        self.result_path = result_path
        self.run_result_folder = os.path.join(self.result_path,
                                              self.model_name, self.run_name)
        self.run_numbers = self.get_run_numbers()

    def get_run_numbers(self) -> List[int]:
        """Scan run result folder for available run numbers."""
        run_numbers: List[int] = []
        folders = os.scandir(self.run_result_folder)
        for entry in folders:
            if entry.is_dir():
                run_numbers.append(int(entry.name))

        run_numbers.sort()
        return run_numbers

    def load_run_utility_df(self, run_number: int) -> pd.DataFrame:
        df_folder = os.path.join(self.run_result_folder, str(run_number),
                                 "single-model-train")

        file_names: List[str] = []
        csv_files = os.scandir(df_folder)
        for entry in csv_files:
            if entry.is_file() and entry.name.endswith(".csv"):
                file_names.append(entry.name)

        # find csv file with longest name -> this is probably our wanted csv file since it includes the most datasets
        df_filename = max(file_names, key=len)
        df_filename = os.path.join(df_folder, df_filename)
        df = pd.read_csv(df_filename, index_col=False)
        return df

    def analyse_utility(self):
        utility_df = self.build_combined_model_utility_df()

        acc_df = utility_df["accuracy"]
        f1_df = utility_df["f1-score"]
        loss_df = utility_df["loss"]

        ###
        # Accuracy
        ###
        acc_vis_filename: str = os.path.join(self.run_result_folder,
                                             "run_accuracy_comparison.png")
        acc_df_filename = os.path.join(self.run_result_folder,
                                       "accuracy_model_comparison.csv")
        acc_fig = self._visualize_df(
            acc_df,
            yLabel="accuracy",
            xLabel="run number",
            title="Model accuracy comparison between mutliple runs",
        )
        print(f"Saving accuracy comparison figure to {acc_vis_filename}")
        acc_fig.savefig(acc_vis_filename)
        save_dataframe(acc_df, acc_df_filename)

        ###
        # F1-Score
        ###
        f1score_df_filename = os.path.join(self.run_result_folder,
                                           "f1score_model_comparison.csv")
        f1score_vis_filename: str = os.path.join(self.run_result_folder,
                                                 "run_f1score_comparison.png")
        f1_fig = self._visualize_df(
            f1_df,
            yLabel="f1-score",
            xLabel="run number",
            title="Model f1-score comparison between mutliple runs",
        )
        print(f"Saving f1-score comparison figure to {f1score_vis_filename}")
        f1_fig.savefig(f1score_vis_filename)
        save_dataframe(f1_df, f1score_df_filename)

        ###
        # Loss
        ###
        loss_df_filename = os.path.join(self.run_result_folder,
                                        "loss_model_comparison.csv")
        loss_vis_filename: str = os.path.join(self.run_result_folder,
                                              "run_loss_comparison.png")
        loss_fig = self._visualize_df(
            loss_df,
            yLabel="loss",
            xLabel="run number",
            title="Model loss comparison between mutliple runs",
        )
        print(f"Saving loss comparison figure to {f1score_vis_filename}")
        loss_fig.savefig(loss_vis_filename)
        save_dataframe(loss_df, loss_df_filename)

    def build_combined_model_utility_df(self) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        for run in self.run_numbers:
            run_df: pd.DataFrame = self.load_run_utility_df(run)
            run_df["run"] = run
            col = run_df["run"]
            run_df.drop(labels=["run"], axis=1, inplace=True)
            run_df.insert(0, "run", col)
            dfs.append(run_df)

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        combined_df = combined_df.pivot(
            index=["name", "type"],
            columns="run",
            values=["accuracy", "f1-score", "loss"],
        )
        averaged = combined_df.groupby("type").mean()
        averaged.rename(index={
            "test": "average test",
            "train": "average train"
        },
                        inplace=True)
        combined_df = pd.concat([combined_df, averaged])
        return combined_df

    def _visualize_df(
        self,
        df: pd.DataFrame,
        xLabel: str,
        yLabel: str,
        title: str,
        use_grid: bool = True,
        use_legend: bool = True,
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots()
        for index, row in df.iterrows():
            if type(index) == "tuple":
                index = " ".join(index)
            ax.plot(range(len(row)), row, label=index)

        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        ax.legend()
        plt.legend(loc=(1.04, 0))
        plt.subplots_adjust(right=0.72)
        ax.grid()
        return fig
