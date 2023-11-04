import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import save_dataframe

pd.options.mode.chained_assignment = None

FIGSIZE = (5, 3)


class UtilityAnalyser:

    def __init__(self, result_path: str, settings: RunSettings):
        self.settings = settings
        self.run_name = self.settings.run_name
        self.model_name = self.settings.model_name
        self.result_path = result_path
        self.run_result_folder = os.path.join(self.result_path,
                                              self.model_name, self.run_name)

        self.x_axis_name = settings.x_axis_name
        self.x_axis_values = settings.x_axis_values

        self.combined_result_folder = os.path.join(
            self.run_result_folder, "utility-analysis-combined")
        check_create_folder(self.combined_result_folder)

        if self.settings.analysis_run_numbers:
            self.run_numbers = self.settings.analysis_run_numbers
        else:
            print("Error: No analysis run number range specified!")
            sys.exit(1)

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
        acc_df, f1_df, loss_df, gap_df = self.build_combined_model_utility_dfs(
            self.settings.analysis_run_numbers)

        acc_df = acc_df.loc[self.run_numbers]
        f1_df = f1_df.loc[self.run_numbers]
        loss_df = loss_df.loc[self.run_numbers]
        gap_df = gap_df.loc[self.run_numbers]

        ###
        # Accuracy
        ###
        acc_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"run_accuracy_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        acc_df_filename = os.path.join(
            self.combined_result_folder,
            f"accuracy_model_comparison_r{''.join(map(str,self.run_numbers))}.csv",
        )
        acc_fig = self._visualize_df(
            acc_df,
            run_range=self.run_numbers,
            yLabel="Accuracy",
            xLabel="Run Number",
            title="Model Accuracy Comparison - Mutliple Runs",
        )
        print(f"Saving accuracy comparison figure to {acc_vis_filename}")
        acc_fig.savefig(acc_vis_filename)
        save_dataframe(acc_df, acc_df_filename)

        ###
        # F1-Score
        ###
        f1score_df_filename = os.path.join(
            self.combined_result_folder,
            f"f1score_model_comparison_r{''.join(map(str,self.run_numbers))}.csv",
        )
        f1score_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"run_f1score_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        f1_fig = self._visualize_df(
            f1_df,
            run_range=self.run_numbers,
            yLabel="F1-Score",
            xLabel="Run Number",
            title="Model F1-Score Comparison - Mutliple Runs",
        )
        print(f"Saving f1-score comparison figure to {f1score_vis_filename}")
        f1_fig.savefig(f1score_vis_filename)
        save_dataframe(f1_df, f1score_df_filename)

        ###
        # Loss
        ###
        loss_df_filename = os.path.join(
            self.combined_result_folder,
            f"loss_model_comparison_r{''.join(map(str,self.run_numbers))}.csv",
        )
        loss_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"run_loss_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        loss_fig = self._visualize_df(
            loss_df,
            run_range=self.run_numbers,
            yLabel="Loss",
            xLabel="Run Number",
            title="Model Loss Comparison - Mutliple Runs",
        )
        print(f"Saving loss comparison figure to {f1score_vis_filename}")
        loss_fig.savefig(loss_vis_filename)
        save_dataframe(loss_df, loss_df_filename)

        ###
        # Train/Test Gap
        ###
        gap_df_filename = os.path.join(
            self.combined_result_folder,
            f"gap_model_comparison_r{''.join(map(str,self.run_numbers))}.csv",
        )
        gap_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"run_gap_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        gap_fig = self._visualize_df(
            gap_df,
            run_range=self.run_numbers,
            yLabel="Train/ Test Gap",
            xLabel="Run Number",
            title="Model Train/ Test Gap Comparison - Mutliple Runs",
        )
        print(f"Saving train/test gap comparison figure to {gap_vis_filename}")
        gap_fig.savefig(gap_vis_filename)
        save_dataframe(gap_df, gap_df_filename)

    def build_combined_model_utility_dfs(
            self, runs: List[int]) -> Tuple[pd.DataFrame]:
        """Return a tuple of 3 dataframes. Each dataframes represents one utility anaylsis from the evaluated model.

        The dataframes have the following order in the tuple:
            1 -> Accuracy
            2 -> F1-Score
            3 -> Loss
            4 -> Train/Test Gap
        """
        combined_utility_df: pd.DataFrame = None

        for run in runs:
            run_df: pd.DataFrame = self.load_run_utility_df(run)
            run_df["run"] = run

            if combined_utility_df is None:
                combined_utility_df = run_df
            else:
                combined_utility_df = pd.concat([combined_utility_df, run_df])

        # this dict holds the name of the dataset as keys (cifar10_c52 and cifar10_30 are just cifar10)
        # to the keys the dict holds a list, one list entry is a tuple of (train value, test value)
        # each list entry represents one experiment
        run_dict_accuracy: Dict[str, List[float]] = defaultdict(list)
        run_dict_f1score: Dict[str, List[float]] = defaultdict(list)
        run_dict_loss: Dict[str, List[float]] = defaultdict(list)
        run_dict_gap: Dict[str, List[float]] = defaultdict(list)

        for i in combined_utility_df.iterrows():
            ds_name = i[1]["name"]

            if "_gray" in ds_name:
                ds_name = ds_name.split("_")[0] + "_gray"
            elif "_" in ds_name:
                ds_name = ds_name.split("_")[0]

            if i[1]["type"] == "train":
                run_dict_accuracy[ds_name + "_train"].append(i[1]["accuracy"])
                run_dict_f1score[ds_name + "_train"].append(i[1]["f1-score"])
                run_dict_loss[ds_name + "_train"].append(i[1]["loss"])
            elif i[1]["type"] == "test":
                run_dict_accuracy[ds_name + "_test"].append(i[1]["accuracy"])
                run_dict_f1score[ds_name + "_test"].append(i[1]["f1-score"])
                run_dict_loss[ds_name + "_test"].append(i[1]["loss"])

            run_dict_gap[ds_name + "_test"].append(i[1]["accuracy"])

        df_acc = pd.DataFrame.from_dict(run_dict_accuracy)
        df_acc["avg_train"] = df_acc.filter(like="_train").mean(axis=1)
        df_acc["avg_test"] = df_acc.filter(like="_test").mean(axis=1)
        df_acc = df_acc.set_axis(runs)

        df_f1 = pd.DataFrame.from_dict(run_dict_f1score)
        df_f1["avg_train"] = df_f1.filter(like="_train").mean(axis=1)
        df_f1["avg_test"] = df_f1.filter(like="_test").mean(axis=1)
        df_f1 = df_f1.set_axis(runs)

        df_loss = pd.DataFrame.from_dict(run_dict_loss)
        df_loss["avg_train"] = df_loss.filter(like="_train").mean(axis=1)
        df_loss["avg_test"] = df_loss.filter(like="_test").mean(axis=1)
        df_loss = df_loss.set_axis(runs)

        for k, acc_list in run_dict_gap.items():
            diff_acc_list = []
            for i in range(0, len(acc_list) - 1, 2):
                diff_acc_list.append(acc_list[i] - acc_list[i + 1])
            run_dict_gap[k] = diff_acc_list

        df_gap = pd.DataFrame.from_dict(run_dict_gap)
        df_gap["avg_test"] = df_gap.filter(like="_test").mean(axis=1)
        df_gap = df_gap.set_axis(runs)

        return (df_acc, df_f1, df_loss, df_gap)

    def _visualize_df(
        self,
        df: pd.DataFrame,
        run_range: List[int],
        xLabel: str,
        yLabel: str,
        title: str,
        use_grid: bool = True,
        use_legend: bool = True,
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for name, values in df.items():
            if name.endswith("_test"):
                name = name.removesuffix("_test")
                if name.startswith("avg"):
                    name = name.replace("avg", "average")
                    if self.x_axis_values is not None:
                        run_range = self.x_axis_values

                    ax.plot(
                        run_range,
                        values,
                        label=name,
                        linestyle="dashed",
                        linewidth=3,
                    )
                else:
                    if "-" in name:
                        name = name.split("-")[0]
                    ax.plot(run_range, values, label=name, marker="x")

        if self.x_axis_name is not None:
            ax.set(xlabel=self.x_axis_name, ylabel=yLabel)
        else:
            ax.set(xlabel=xLabel, ylabel=yLabel)
        ax.legend()
        plt.xticks(run_range)
        # plt.yticks(np.arange(0, 1, 0.1))
        plt.legend()
        ax.grid()
        return fig
