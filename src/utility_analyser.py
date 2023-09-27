import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ppml_datasets.utils import check_create_folder
from settings import RunSettings
from util import get_run_numbers, save_dataframe

pd.options.mode.chained_assignment = None


class UtilityAnalyser:

    def __init__(self, result_path: str, settings: RunSettings):
        self.settings = settings
        self.run_name = self.settings.run_name
        self.model_name = self.settings.model_name
        self.result_path = result_path
        self.run_result_folder = os.path.join(self.result_path,
                                              self.model_name, self.run_name)

        self.combined_result_folder = os.path.join(
            self.run_result_folder, "utility-analysis-combined")
        check_create_folder(self.combined_result_folder)

        if self.settings.analysis_run_numbers:
            self.run_numbers = self.settings.analysis_run_numbers
        else:
            self.run_numbers = get_run_numbers(self.run_result_folder)

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
        acc_df, f1_df, loss_df = self.build_combined_model_utility_dfs()

        acc_df = acc_df.loc[self.run_numbers]
        f1_df = f1_df.loc[self.run_numbers]
        loss_df = loss_df.loc[self.run_numbers]

        ###
        # Accuracy
        ###
        acc_vis_filename: str = os.path.join(
            self.combined_result_folder,
            f"run_accuracy_comparison_r{''.join(map(str,self.run_numbers))}.png",
        )
        acc_df_filename = os.path.join(
            self.combined_result_folder,
            f"accuracy_model_comparison_r{''.join(map(str,self.run_numbers))}",
        )
        acc_fig = self._visualize_df(
            acc_df,
            run_range=self.run_numbers,
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
            yLabel="loss",
            xLabel="run number",
            title="Model loss comparison between mutliple runs",
        )
        print(f"Saving loss comparison figure to {f1score_vis_filename}")
        loss_fig.savefig(loss_vis_filename)
        save_dataframe(loss_df, loss_df_filename)

    def build_combined_model_utility_dfs(self) -> Tuple[pd.DataFrame]:
        """Return a tuple of 3 dataframes. Each dataframes represents one utility anaylsis from the evaluated model.

        The dataframes have the following order in the tuple:
            1 -> Accuracy
            2 -> F1-Score
            3 -> Loss
        """
        # here we have a bug in the program
        # every utility df from the '-em' cli parameter saves the complete utility values from all previous runs
        # so loading only the last run's utility df is enough to get all utility data
        last_run = self.run_numbers[-1]
        run_df: pd.DataFrame = self.load_run_utility_df(last_run)

        # this dict holds the name of the dataset as keys (cifar10_c52 and cifar10_30 are just cifar10)
        # to the keys the dict holds a list, one list entry is a tuple of (train value, test value)
        # each list entry represents one experiment
        run_dict_accuracy: Dict[str, List[float]] = defaultdict(list)
        run_dict_f1score: Dict[str, List[float]] = defaultdict(list)
        run_dict_loss: Dict[str, List[float]] = defaultdict(list)

        for i in run_df.iterrows():
            ds_name = i[1]["name"]
            if "_" in ds_name:
                ds_base_name = ds_name.split("_")[0]
                if i[1]["type"] == "train":
                    run_dict_accuracy[ds_base_name + "_train"].append(
                        i[1]["accuracy"])
                    run_dict_f1score[ds_base_name + "_train"].append(
                        i[1]["f1-score"])
                    run_dict_loss[ds_base_name + "_train"].append(i[1]["loss"])
                elif i[1]["type"] == "test":
                    run_dict_accuracy[ds_base_name + "_test"].append(
                        i[1]["accuracy"])
                    run_dict_f1score[ds_base_name + "_test"].append(
                        i[1]["f1-score"])
                    run_dict_loss[ds_base_name + "_test"].append(i[1]["loss"])

        df_acc = pd.DataFrame.from_dict(run_dict_accuracy)
        df_acc["avg_train"] = df_acc.filter(like="_train").mean(axis=1)
        df_acc["avg_test"] = df_acc.filter(like="_test").mean(axis=1)

        df_f1 = pd.DataFrame.from_dict(run_dict_f1score)
        df_f1["avg_train"] = df_f1.filter(like="_train").mean(axis=1)
        df_f1["avg_test"] = df_f1.filter(like="_test").mean(axis=1)

        df_loss = pd.DataFrame.from_dict(run_dict_loss)
        df_loss["avg_train"] = df_loss.filter(like="_train").mean(axis=1)
        df_loss["avg_test"] = df_loss.filter(like="_test").mean(axis=1)

        return (df_acc, df_f1, df_loss)

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
        fig, ax = plt.subplots(figsize=(10, 10))
        for name, values in df.items():
            # dont include the average values in the graph
            if name.startswith("avg"):
                ax.plot(
                    run_range,
                    values,
                    label=name,
                    linestyle="dotted",
                    linewidth=2,
                )
            else:
                ax.plot(run_range, values, label=name)

        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        ax.legend()
        plt.legend(loc=(1.04, 0))
        plt.subplots_adjust(right=0.72)
        ax.grid()
        return fig
