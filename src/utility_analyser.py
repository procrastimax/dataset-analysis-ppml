import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import save_dataframe

pd.options.mode.chained_assignment = None

FIGSIZE = (5, 3)
AXHLINE_COLOR = "tab:gray"
AXHLINE_WIDTH = 1.0
AXHLINE_STYLE = "-"
LEGEND_ALPHA = 0.5


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
        (
            acc_df,
            f1_df,
            loss_df,
            gap_df,
            class_wise_f1_df,
        ) = self.build_combined_model_utility_dfs(
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
            default_value_indicator=0.5,
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
            default_value_indicator=0.5,
        )
        print(f"Saving f1-score comparison figure to {f1score_vis_filename}")
        f1_fig.savefig(f1score_vis_filename)
        save_dataframe(f1_df, f1score_df_filename)

        f1score_bar_chart_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"bar_chart_f1score_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        f1_bar_chart_fig = self.visualize_df_bar_chart(
            f1_df,
            run_range=self.run_numbers,
            yLabel="F1-Score",
            xLabel="Class",
            default_value_indicator=0.5,
        )
        print(
            f"Saving f1-score bar chart comparison figure to {f1score_bar_chart_vis_filename}"
        )
        f1_bar_chart_fig.savefig(f1score_bar_chart_vis_filename)

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
        )
        print(f"Saving train/test gap comparison figure to {gap_vis_filename}")
        gap_fig.savefig(gap_vis_filename)
        save_dataframe(gap_df, gap_df_filename)

        if class_wise_f1_df is not None:
            ###
            # Class Wise F1
            # calculate average f1 scores for every class and visalize these averages over the runs
            ###
            max_run = class_wise_f1_df["Run"].max()
            # only use the _test datasets
            class_wise_f1_df_test = class_wise_f1_df[
                class_wise_f1_df["Dataset"].str.endswith("_test")]

            # a dict to contain a series of f1-scores for every class
            class_wise_f1_dict: Dict[int, List[float]] = defaultdict(list)

            # iterate over all runs
            for i in range(max_run + 1):
                class_avg = class_wise_f1_df_test.loc[class_wise_f1_df["Run"]
                                                      == i].mean()
                for class_num, j in enumerate(class_avg):
                    if class_num > 0:
                        class_wise_f1_dict[class_num - 1].append(j)

            x_values = self.run_numbers
            if self.settings.x_axis_values is not None:
                x_values = self.settings.x_axis_values

            fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

            for class_num, f1scores in class_wise_f1_dict.items():
                ax.plot(x_values,
                        f1scores,
                        label=f"Class {class_num}",
                        marker="x")
            ax.set(xlabel=self.x_axis_name, ylabel="F1-Score")
            plt.legend(
                loc="lower left",
                labelspacing=0.4,
                columnspacing=1,
                framealpha=0.5,
                handlelength=1.2,
                handletextpad=0.3,
                ncols=3,
                fontsize="small",
                markerscale=0.8,
            )
            ax.axhline(
                y=0.5,
                linestyle=AXHLINE_STYLE,
                color=AXHLINE_COLOR,
                linewidth=AXHLINE_WIDTH,
            )
            ax.grid()
            plt.xticks(self.run_numbers)
            class_wise_f1_df_filename = os.path.join(
                self.combined_result_folder,
                f"class_wise_f1_r{''.join(map(str,self.run_numbers))}.csv",
            )
            class_wise_f1_vis_filename: str = os.path.join(
                self.combined_result_folder,
                f"class_wise_f1_r{''.join(map(str,self.run_numbers))}.png",
            )
            print(
                f"Saving class wise f1-scores figure to {class_wise_f1_vis_filename}"
            )
            plt.savefig(class_wise_f1_vis_filename)
            save_dataframe(class_wise_f1_df, class_wise_f1_df_filename)

            # --------
            # create for every run a grouped bar chart to show the F1-scores of each class
            # --------
            fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
            # iterate over all runs
            for idx in range(max_run + 1):
                run_f1_scores = class_wise_f1_df_test.loc[
                    class_wise_f1_df["Run"] == idx]

                class_wise_value_dict: Dict[str,
                                            List[float]] = defaultdict(list)

                for i in run_f1_scores.iterrows():
                    ds_name = i[1]["Dataset"]
                    for j in i[1][2:]:
                        class_wise_value_dict[ds_name].append(j)

                x = np.arange(len(list(class_wise_value_dict.values())[0]))
                width = 0.22
                multiplier = 0

                name_list = []

                fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
                for ds_name, values in class_wise_value_dict.items():
                    ds_name = ds_name.removesuffix("_test")
                    name_list.append(ds_name)

                    offset = width * multiplier
                    ax.bar(x + offset, values, width, label=ds_name)
                    multiplier += 1

                ax.axhline(
                    y=0.5,
                    linestyle=AXHLINE_STYLE,
                    color=AXHLINE_COLOR,
                    linewidth=AXHLINE_WIDTH,
                )
                ax.set_ylabel("F1-Score")
                ax.set_xlabel("Class")
                ax.set_xticks(x + width, x)
                ax.legend(loc="lower right")

                bar_chart_class_wise_f1_vis_filename: str = os.path.join(
                    self.combined_result_folder,
                    f"bar_chart_class_wise_f1_{''.join(name_list)}_r{idx}.png",
                )
                print(
                    f"Saving class wise bar chart f1-scores figure to {bar_chart_class_wise_f1_vis_filename}"
                )
                plt.savefig(bar_chart_class_wise_f1_vis_filename)

        plt.close()

    def build_combined_model_utility_dfs(
            self, runs: List[int]) -> Tuple[pd.DataFrame]:
        """Return a tuple of 3 dataframes. Each dataframes represents one utility anaylsis from the evaluated model.

        The dataframes have the following order in the tuple:
            1 -> Accuracy
            2 -> F1-Score
            3 -> Loss
            4 -> Train/Test Gap
            5 -> Class Wise F1 Scores
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

        run_dict_class_f1: Dict[str, List[Dict[str,
                                               float]]] = defaultdict(list)

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

                if "class-wise" in i[1]:
                    if not pd.isna(i[1]["class-wise"]):
                        # we have to replace the '' with "" in order to parse it with the json module
                        class_wise_str_dict = i[1]["class-wise"].replace(
                            "'", '"')
                        run_dict_class_f1[ds_name + "_train"].append(
                            json.loads(class_wise_str_dict))

            elif i[1]["type"] == "test":
                run_dict_accuracy[ds_name + "_test"].append(i[1]["accuracy"])
                run_dict_f1score[ds_name + "_test"].append(i[1]["f1-score"])
                run_dict_loss[ds_name + "_test"].append(i[1]["loss"])

                if "class-wise" in i[1]:
                    if not pd.isna(i[1]["class-wise"]):
                        class_wise_str_dict = i[1]["class-wise"].replace(
                            "'", '"')
                        run_dict_class_f1[ds_name + "_test"].append(
                            json.loads(class_wise_str_dict))

            # add all accuracy values either test or train to the list to later calculate the difference
            run_dict_gap[ds_name + "_test"].append(i[1]["accuracy"])

        # parse dict of class wise f1 to dataframe -> the dataframe looks like this but for all runs
        """
            Dataset          Run    0         1         2         3         4         5         6         7         8         9
        0  cifar10_train       4    0.9996  0.999667  0.998499  0.997999  0.998334  0.997798    0.9985  0.997667  0.997996  0.998997
        1   cifar10_test       4  0.739949  0.777311  0.667954   0.56614  0.694988  0.606215  0.782974  0.741485  0.756571  0.598794
        2   fmnist_train       4  0.998601  0.999889  0.995746  0.998572  0.994177    0.9998  0.992209       1.0       1.0       1.0
        3    fmnist_test       4  0.865911  0.989495  0.867998  0.922388  0.861249  0.973645  0.746549  0.949466  0.978294  0.956208
        4    mnist_train       4    0.9999       1.0  0.999875       1.0       1.0       1.0    0.9995       1.0       1.0       1.0
        5     mnist_test       4  0.989879  0.992105  0.980373  0.987715  0.985405  0.977553  0.985248  0.983366  0.989158  0.972617
        6     svhn_train       4       1.0  0.998701  0.999444  0.998122  0.998709   0.99799  0.998588   0.99924   0.99694  0.996778
        7      svhn_test       4  0.861136  0.915769  0.877148  0.804523  0.871628  0.836205  0.783949  0.886783  0.804243  0.780851"
        """
        largest_class_count = None
        # check if dict is empty -> if so, dont create class wise f1 df
        if run_dict_class_f1:
            cols = list(list(run_dict_class_f1.values())[0][0].keys())
            cols.insert(0, "Run")
            cols.insert(0, "Dataset")
            df_class_f1 = pd.DataFrame(columns=cols,
                                       index=range(
                                           len(runs) * len(run_dict_class_f1)))

            num_idx = 0
            for ds_name, class_wise_f1_list in run_dict_class_f1.items():
                for i, class_wise_f1 in enumerate(class_wise_f1_list):
                    # this is for the case of class the class count experiment
                    # there the number of classes reduces, so we have to fill the value array with values
                    if largest_class_count is None:
                        largest_class_count = len(class_wise_f1.values())

                    if len(class_wise_f1.values()) < largest_class_count:
                        padded_class_wise_value_list = list(class_wise_f1.values())
                        padded_class_wise_value_list += [0.0] * (largest_class_count - len(padded_class_wise_value_list))
                        print(padded_class_wise_value_list)
                        df_class_f1.iloc[num_idx] = [ds_name, i] + padded_class_wise_value_list
                    else:
                        df_class_f1.iloc[num_idx] = [ds_name, i] + list(
                            class_wise_f1.values())
                    num_idx += 1
        else:
            df_class_f1 = None

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

        return (df_acc, df_f1, df_loss, df_gap, df_class_f1)

    def visualize_df_bar_chart(
        self,
        df: pd.DataFrame,
        run_range: List[int],
        xLabel: str,
        yLabel: str,
        use_grid: bool = True,
        use_legend: bool = True,
        default_value_indicator: Optional[float] = None,
    ) -> matplotlib.figure.Figure:
        x = np.arange(len(run_range))
        width = 0.22
        multiplier = 0

        if self.x_axis_values is not None:
            run_range = self.x_axis_values

        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for name, values in df.items():
            if name.endswith("_test"):
                name = name.removesuffix("_test")
                if name.startswith("avg"):
                    continue

                offset = width * multiplier
                ax.bar(x + offset, values, width, label=name)
                multiplier += 1

        ax.set_ylabel(yLabel)
        if self.x_axis_name is not None:
            ax.set_xlabel(self.x_axis_name)
        else:
            ax.set_xlabel(xLabel)
        ax.set_xticks(x + width, run_range)

        # draw a fine line to indicate default values (like 0.5 in accuracy)
        if default_value_indicator is not None:
            ax.axhline(
                y=default_value_indicator,
                linestyle=AXHLINE_STYLE,
                color=AXHLINE_COLOR,
                linewidth=AXHLINE_WIDTH,
            )

        ax.legend(loc="lower right")
        return fig

    def _visualize_df(
        self,
        df: pd.DataFrame,
        run_range: List[int],
        xLabel: str,
        yLabel: str,
        use_grid: bool = True,
        use_legend: bool = True,
        default_value_indicator: Optional[float] = None,
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        for name, values in df.items():
            if self.x_axis_values is not None:
                run_range = self.x_axis_values
            if name.endswith("_test"):
                name = name.removesuffix("_test")
                if name.startswith("avg"):
                    name = name.replace("avg", "average")

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

        # draw a fine line to indicate default values (like 0.5 in accuracy)
        if default_value_indicator is not None:
            ax.axhline(
                y=default_value_indicator,
                linestyle=AXHLINE_STYLE,
                color=AXHLINE_COLOR,
                linewidth=AXHLINE_WIDTH,
            )

        if self.x_axis_name is not None:
            ax.set(xlabel=self.x_axis_name, ylabel=yLabel)
        else:
            ax.set(xlabel=xLabel, ylabel=yLabel)
        ax.legend()
        ax.set_xticks(run_range)
        # plt.yticks(np.arange(0, 1, 0.1))
        plt.legend()
        ax.grid()
        return fig
