import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResults, SingleAttackResult
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from typing import Optional, List, Dict, Tuple
import os

from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import check_create_folder

from util import pickle_object, save_dataframe, find_nearest
from datasetstore import DatasetStore


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
        self.attack_statistics_folder: str = os.path.join(self.amia_result_path, "attack-statistics")
        self.single_model_train_results: str = os.path.join(self.amia_result_path, "single-model-train")

        self.attack_statistics_folder_combined: str = os.path.join(self.attack_statistics_folder, "combined")
        check_create_folder(self.attack_statistics_folder_combined)

        self.dataset_data: Dict[str, DatasetStore] = {}

        for ds in ds_list:
            model_save_path: str = os.path.join(model_path, str(run_number), ds.dataset_name)
            shadow_model_save_path: str = os.path.join(model_path, str(run_number), "shadow_models", ds.dataset_name)
            numpy_path: str = os.path.join(shadow_model_save_path, "data")
            in_indices_filename = os.path.join(numpy_path, "in_indices.pckl")
            stat_filename = os.path.join(numpy_path, "model_stat.pckl")
            loss_filename = os.path.join(numpy_path, "model_loss.pckl")

            attack_result_list_filename = os.path.join(self.attack_statistics_folder, "pickles", f"{ds.dataset_name}_attack_results.pckl")
            attack_baseline_result_list_filename = os.path.join(self.attack_statistics_folder, "pickles", f"{ds.dataset_name}_attack_baseline_results.pckl")

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
            self.dataset_data[ds_name].attack_result_df = ds_store.calculate_tpr_at_fixed_fpr(ds_store.attack_result_list, attack_name="amia")
            self.dataset_data[ds_name].attack_baseline_result_df = ds_store.calculate_tpr_at_fixed_fpr(ds_store.attack_baseline_result_list, attack_name="mia")

            df_amia_filename = os.path.join(self.attack_statistics_folder, f"amia_attack_statistic_results_{ds_store.ds_name}.csv")
            df_mia_filename = os.path.join(self.attack_statistics_folder, f"mia_attack_statistic_results_{ds_store.ds_name}.csv")

            save_dataframe(ds_store.attack_result_df, df_amia_filename)
            save_dataframe(ds_store.attack_baseline_result_df, df_mia_filename)

            ds_store.create_all_in_one_roc_curve()
            mean_tpr, mean_fpr = ds_store.create_average_roc_curve(attack_result_list=ds_store.attack_result_list, generate_all_rocs=True, generate_std_area=True)
            ds_store.create_average_roc_curve(attack_result_list=ds_store.attack_result_list, generate_all_rocs=True, generate_std_area=False)
            ds_store.create_average_roc_curve(attack_result_list=ds_store.attack_result_list, generate_all_rocs=False, generate_std_area=True)
            ds_store.create_average_roc_curve(attack_result_list=ds_store.attack_result_list, generate_all_rocs=False, generate_std_area=False)
            ds_store.create_average_roc_curve(attack_result_list=ds_store.attack_baseline_result_list, name="MIA", generate_all_rocs=True, generate_std_area=True)
            ds_store.create_mia_vs_amia_roc_curves()

            self.dataset_data[ds_name].mean_tpr = mean_tpr
            self.dataset_data[ds_name].mean_fpr = mean_fpr

        self.create_combined_averaged_roc_curve(self.dataset_data.values())

    def create_combined_averaged_roc_curve(self, ds_stores: List[DatasetStore]):
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], 'r--', lw=1.0)

        name_list = []

        for store in ds_stores:
            (idx_0001, _) = find_nearest(store.mean_tpr, 0.001)
            (idx_01, _) = find_nearest(store.mean_tpr, 0.1)
            name_list.append(store.ds_name)

            avg_auc = metrics.auc(store.mean_fpr, store.mean_tpr)
            ax.plot(store.mean_fpr, store.mean_tpr, label=f"{store.ds_name}\nAUC:{avg_auc:.3f} FPR@0.1:{store.mean_fpr[idx_01]:.3f} FPR@0.001:{store.mean_fpr[idx_0001]:.4f}")

        ax.set(xlabel="TPR", ylabel="FPR")
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text(f"Receiver Operator Characteristics Averaged - {' '.join(name_list)}")
        plt.xlim([0.00001, 1])
        plt.ylim([0.00001, 1])
        plt.legend()

        plt_name = os.path.join(self.attack_statistics_folder_combined, f"averaged_roc_curve_{'-'.join(name_list)}_advanced_mia_results.png")
        plt.savefig(plt_name)
        print(f"Saved all-in-one ROC curve {plt_name}")
        plt.close()
