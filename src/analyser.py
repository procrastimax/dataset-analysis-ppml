from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults, SingleAttackResult
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from typing import List, Dict
import os

from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder

from util import save_dataframe, find_nearest
from datasetstore import DatasetStore


class Analyser():
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
            model_save_path: str = os.path.join(model_path, model_name, run_name, str(run_number), ds.dataset_name)
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
