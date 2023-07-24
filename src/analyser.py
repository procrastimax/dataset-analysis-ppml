from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults, SingleAttackResult
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn import metrics
from typing import List, Dict
import os

from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder

from util import save_dataframe, find_nearest
from datasetstore import DatasetStore


class AttackAnalyser():
    def __init__(self,
                 ds_list: List[AbstractDataset],
                 model_name: str,
                 run_name: str,
                 run_number: int,
                 result_path: str,
                 model_path: str,
                 num_shadow_models: int,
                 include_mia: bool = False,
                 ):
        self.result_path = result_path
        check_create_folder(self.result_path)

        self.include_mia = include_mia

        self.ds_list = ds_list
        self.num_shadow_models = num_shadow_models

        self.amia_result_path = os.path.join(result_path, model_name, run_name, str(run_number))
        self.attack_statistics_folder: str = os.path.join(
            self.amia_result_path, "attack-statistics")
        self.single_model_train_results: str = os.path.join(
            self.amia_result_path, "single-model-train")

        self.attack_statistics_folder_combined: str = os.path.join(
            self.attack_statistics_folder, "combined")
        check_create_folder(self.attack_statistics_folder_combined)

        self.dataset_data: Dict[str, DatasetStore] = {}

        for ds in ds_list:
            model_save_path: str = os.path.join(
                model_path, model_name, run_name, str(run_number), ds.dataset_name)
            shadow_model_save_path: str = os.path.join(
                model_path, model_name, run_name, str(run_number), "shadow_models", ds.dataset_name)
            numpy_path: str = os.path.join(shadow_model_save_path, "data")
            in_indices_filename = os.path.join(numpy_path, "in_indices.pckl")
            stat_filename = os.path.join(numpy_path, "model_stat.pckl")
            loss_filename = os.path.join(numpy_path, "model_loss.pckl")

            attack_result_list_filename = os.path.join(
                self.attack_statistics_folder, "pickles", f"{ds.dataset_name}_attack_results.pckl")
            attack_baseline_result_list_filename = os.path.join(
                self.attack_statistics_folder, "pickles", f"{ds.dataset_name}_attack_baseline_results.pckl")

            ds_store = DatasetStore(shadow_model_dir=shadow_model_save_path,
                                    model_save_path=model_save_path,
                                    ds_name=ds.dataset_name,
                                    numpy_path=numpy_path,
                                    in_indices_filename=in_indices_filename,
                                    stat_filename=stat_filename,
                                    loss_filename=loss_filename,
                                    attack_result_list_filename=attack_result_list_filename,
                                    attack_baseline_result_list_filename=attack_baseline_result_list_filename,
                                    attack_result_folder=self.attack_statistics_folder,
                                    )
            self.dataset_data[ds.dataset_name] = ds_store

    def generate_results(self):
        for (ds_name, ds_store) in self.dataset_data.items():
            ds_store.load_saved_values()

            self.dataset_data[ds_name].attack_result_df = ds_store.create_complete_dataframe(
                ds_store.attack_result_list, attack_name="amia")
            self.dataset_data[ds_name].set_best_attack_run_idx(
                self.dataset_data[ds_name].attack_result_df)

            df_amia_filename = os.path.join(
                self.attack_statistics_folder, f"amia_attack_statistic_results_{ds_store.ds_name}.csv")
            save_dataframe(ds_store.attack_result_df, df_amia_filename)

            if self.include_mia:
                self.dataset_data[ds_name].attack_baseline_result_df = ds_store.create_complete_dataframe(
                    ds_store.attack_baseline_result_list, attack_name="mia")
                df_mia_filename = os.path.join(
                    self.attack_statistics_folder, f"mia_attack_statistic_results_{ds_store.ds_name}.csv")
                save_dataframe(ds_store.attack_baseline_result_df, df_mia_filename)

            ds_store.create_all_in_one_roc_curve()
            mean_tpr, mean_fpr = ds_store.create_average_roc_curve(
                attack_result_list=ds_store.attack_result_list, generate_all_rocs=True, generate_std_area=True)
            self.dataset_data[ds_name].mean_tpr = mean_tpr
            self.dataset_data[ds_name].mean_fpr = mean_fpr
            ds_store.create_average_roc_curve(
                attack_result_list=ds_store.attack_result_list, generate_all_rocs=True, generate_std_area=False)
            ds_store.create_average_roc_curve(
                attack_result_list=ds_store.attack_result_list, generate_all_rocs=False, generate_std_area=True)
            ds_store.create_average_roc_curve(
                attack_result_list=ds_store.attack_result_list, generate_all_rocs=False, generate_std_area=False)

            if self.include_mia:
                ds_store.create_average_roc_curve(
                    attack_result_list=ds_store.attack_baseline_result_list, name="MIA", generate_all_rocs=True, generate_std_area=True)
                ds_store.create_mia_vs_amia_roc_curves()

        self.create_combined_best_run_fpr0001(list(self.dataset_data.values()))
        self.create_combined_best_run_fpr01(list(self.dataset_data.values()))
        self.create_combined_best_run_auc(list(self.dataset_data.values()))
        self.create_combined_averaged_roc_curve(list(self.dataset_data.values()))
        self.create_combined_df(list(self.dataset_data.values()))

    def create_combined_df(self, ds_stores: List[DatasetStore]):
        combined_list = []
        columns = ["name", "type"]
        columns.extend(ds_stores[0].attack_result_df.tail(
            4).select_dtypes(include=np.number).columns.tolist())

        ds_names = []

        for store in ds_stores:
            ds_list = []
            ds_names.append(store.ds_name)

            best_list = store.attack_result_df.select_dtypes(
                include=np.number).iloc[store.best_idx_auc].values.tolist()
            tmp_list = [store.ds_name, "best_auc"]
            tmp_list.extend(best_list)
            ds_list.append(tmp_list)

            best_list = store.attack_result_df.select_dtypes(
                include=np.number).iloc[store.best_idx_fpr01].values.tolist()
            tmp_list = [store.ds_name, "best_fpr01"]
            tmp_list.extend(best_list)
            ds_list.append(tmp_list)

            best_list = store.attack_result_df.select_dtypes(
                include=np.number).iloc[store.best_idx_fpr0001].values.tolist()
            tmp_list = [store.ds_name, "best_idx_fpr0001"]
            tmp_list.extend(best_list)
            ds_list.append(tmp_list)

            for i, val in enumerate(store.attack_result_df.tail(4).select_dtypes(include=np.number).values.tolist()):
                # this is nasty, but I'm too lazy to properly handle pandas 1.3.4 - hopefully this does not backfire
                num_type = ""
                if i == 0:
                    num_type = "mean"
                elif i == 1:
                    num_type = "min"
                elif i == 2:
                    num_type = "max"
                elif i == 3:
                    num_type = "var"

                tmp_list = [store.ds_name, num_type]
                tmp_list.extend(val)
                ds_list.append(tmp_list)
            combined_list.extend(ds_list)

        combined_df = pd.DataFrame(combined_list, columns=columns)
        print("Combined DF:")
        print(combined_df)
        file_name = os.path.join(self.attack_statistics_folder_combined,
                                 f"combined_df_{'_'.join(ds_names)}.csv")
        save_dataframe(combined_df, filename=file_name)

    def create_combined_averaged_roc_curve(self, ds_stores: List[DatasetStore]):
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0)

        name_list = []

        for store in ds_stores:
            (idx_0001, _) = find_nearest(store.mean_fpr, 0.001)
            (idx_01, _) = find_nearest(store.mean_fpr, 0.1)
            name_list.append(store.ds_name)

            avg_auc = metrics.auc(store.mean_fpr, store.mean_tpr)
            # ax.plot(store.mean_fpr, store.mean_tpr, label=f"{store.ds_name}\nAUC    FPR@0.1  FPR@0.001\n{avg_auc:.3f}  {store.mean_tpr[idx_01]:.4f}   {store.mean_tpr[idx_0001]:.4f}")
            ax.plot(store.mean_fpr, store.mean_tpr, label=f"{store.ds_name} AUC={avg_auc:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text("Receiver Operator Characteristics Averaged")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend(prop={'family': 'DejaVu Sans Mono'})

        plt_name = os.path.join(self.attack_statistics_folder_combined,
                                f"averaged_roc_curve_{'-'.join(name_list)}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr0001(self, ds_stores: List[DatasetStore]):
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0)

        name_list = []

        # sort by fpr01
        ds_stores.sort(key=lambda x: x.get_fpr_at_fixed_tpr(
            x.attack_result_list[x.best_idx_auc].single_attack_results[0])[1], reverse=True)

        for store in ds_stores:
            name_list.append(store.ds_name)

            single_attack: SingleAttackResult = store.attack_result_list[
                store.best_idx_fpr0001].single_attack_results[0]
            fpr_01, fpr_0001 = store.get_fpr_at_fixed_tpr(single_attack)
            fpr = single_attack.roc_curve.fpr
            tpr = single_attack.roc_curve.tpr
            ax.plot(fpr, tpr, label=f"{store.ds_name} FPR@0.001={fpr_0001:.4f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text("ROC Combined Best Run FPR@0.001")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(self.attack_statistics_folder_combined,
                                f"combined_best_run_fpr0001_{'-'.join(name_list)}_results.png")
        plt.savefig(plt_name)
        print(f"Saved combined best attackrun fpr0001 ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_fpr01(self, ds_stores: List[DatasetStore]):
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0)

        name_list = []

        # sort by fpr01
        ds_stores.sort(key=lambda x: x.get_fpr_at_fixed_tpr(
            x.attack_result_list[x.best_idx_auc].single_attack_results[0])[0], reverse=True)

        for store in ds_stores:
            name_list.append(store.ds_name)

            single_attack: SingleAttackResult = store.attack_result_list[
                store.best_idx_fpr01].single_attack_results[0]
            fpr_01, fpr_0001 = store.get_fpr_at_fixed_tpr(single_attack)
            fpr = single_attack.roc_curve.fpr
            tpr = single_attack.roc_curve.tpr
            ax.plot(fpr, tpr, label=f"{store.ds_name} FPR@0.1={fpr_01:.4f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text("ROC Combined Best Run FPR@0.1")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(self.attack_statistics_folder_combined,
                                f"combined_best_run_fpr01_{'-'.join(name_list)}_results.png")
        plt.savefig(plt_name)
        print(f"Saved combined best attackrun fpr01 ROC curve {plt_name}")
        plt.close()

    def create_combined_best_run_auc(self, ds_stores: List[DatasetStore]):
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.0)

        name_list = []

        # sort by auc
        ds_stores.sort(
            key=lambda x: x.attack_result_list[x.best_idx_auc].single_attack_results[0].roc_curve.get_auc(), reverse=True)

        for store in ds_stores:
            name_list.append(store.ds_name)

            single_attack: SingleAttackResult = store.attack_result_list[store.best_idx_auc].single_attack_results[0]
            fpr = single_attack.roc_curve.fpr
            tpr = single_attack.roc_curve.tpr
            auc = single_attack.roc_curve.get_auc()
            ax.plot(fpr, tpr, label=f"{store.ds_name} AUC={auc:.3f}")

        ax.set(xlabel="FPR", ylabel="TPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text("ROC Combined Best Run AUC")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(self.attack_statistics_folder_combined,
                                f"combined_best_run_auc_{'-'.join(name_list)}_results.png")
        plt.savefig(plt_name)
        print(f"Saved combined best attackrun AUC ROC curve {plt_name}")
        plt.close()


class UtilityAnalyser():
    def __init__(self, result_path: str, run_name: str, model_name: str):
        self.run_name = run_name
        self.model_name = model_name
        self.result_path = result_path
        self.run_result_folder = os.path.join(self.result_path, self.model_name, self.run_name)
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
        df_folder = os.path.join(self.run_result_folder, str(run_number), "single-model-train")

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

        plt.legend(loc=(1.04, 0))
        plt.subplots_adjust(right=0.7)

        ###
        # Accuracy
        ###
        acc_vis_filename: str = os.path.join(self.run_result_folder, "run_accuracy_comparison.png")
        acc_df_filename = os.path.join(self.run_result_folder, "accuracy_model_comparison.csv")
        acc_fig = self._visualize_df(acc_df, xLabel="accuracy", yLabel="run number",
                                     title="Model accuracy comparison between mutliple runs")
        print(f"Saving accuracy comparison figure to {acc_vis_filename}")
        acc_fig.savefig(acc_vis_filename)
        save_dataframe(acc_df, acc_df_filename)

        ###
        # F1-Score
        ###
        f1score_df_filename = os.path.join(self.run_result_folder, "f1score_model_comparison.csv")
        f1score_vis_filename: str = os.path.join(
            self.run_result_folder, "run_f1score_comparison.png")
        f1_fig = self._visualize_df(f1_df, xLabel="f1-score", yLabel="run number",
                                    title="Model f1-score comparison between mutliple runs")
        print(f"Saving f1-score comparison figure to {f1score_vis_filename}")
        f1_fig.savefig(f1score_vis_filename)
        save_dataframe(f1_df, f1score_df_filename)

        ###
        # Loss
        ###
        loss_df_filename = os.path.join(self.run_result_folder, "loss_model_comparison.csv")
        loss_vis_filename: str = os.path.join(
            self.run_result_folder, "run_loss_comparison.png")
        loss_fig = self._visualize_df(loss_df, xLabel="loss", yLabel="run number",
                                      title="Model loss comparison between mutliple runs")
        print(f"Saving loss comparison figure to {f1score_vis_filename}")
        loss_fig.savefig(loss_vis_filename)
        save_dataframe(loss_df, loss_df_filename)

    def build_combined_model_utility_df(self) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        for run in self.run_numbers:
            run_df: pd.DataFrame = self.load_run_utility_df(run)
            run_df["run"] = run
            col = run_df["run"]
            run_df.drop(labels=['run'], axis=1, inplace=True)
            run_df.insert(0, "run", col)
            dfs.append(run_df)

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        combined_df = combined_df.pivot(index=["name", "type"],
                                        columns="run",
                                        values=["accuracy", "f1-score", "loss"])
        averaged = combined_df.groupby("type").mean()
        averaged.rename(index={'test': 'average test',
                               'train': 'average train'}, inplace=True)
        combined_df = pd.concat([combined_df, averaged])
        return combined_df

    def _visualize_df(self, df: pd.DataFrame, xLabel: str, yLabel: str, title: str, use_grid: bool = True, use_legend: bool = True) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots()
        for index, row in df.iterrows():
            if type(index) == "tuple":
                index = " ".join(index)
            ax.plot(range(len(row)), row, label=index)

        ax.set(xlabel=xLabel,
               ylabel=yLabel,
               title=title)
        ax.legend()
        ax.grid()
        return fig
