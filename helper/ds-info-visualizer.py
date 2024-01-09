import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from collections import defaultdict

plt.rcParams["figure.figsize"] = [5, 3]
plt.rcParams["figure.autolayout"] = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--files",
    type=str,
    nargs="+",
    required=True,
    help="List of/ Single Files to be used for creating visualizations. Must be one of the ds-info.json files.",
)

parser.add_argument(
    "-p",
    "--parameter",
    type=str,
    help="Parameter from the ds-info.json files to be used for visualization.",
)

parser.add_argument(
    "-c",
    "--is-class-wise",
    action="store_true",
    help="Flag to indicate if the used '-p' parameter specifies a class-wise attribute. The kind of visualization changes if this is the case.",
)

parser.add_argument(
    "--yName",
    type=str,
    help="Name of the Y-Axis for the generated figure.",
)

args = parser.parse_args()

file_names = args.files
info_key = args.parameter
is_class_wise = args.is_class_wise
y_name = args.yName

# sort the filenames so the order is the same as is the code
file_names.sort()

# get current script dir
curr_dir = os.path.abspath(os.getcwd())

for i, f in enumerate(file_names):
    file_names[i] = os.path.join(curr_dir, f)

print(f"Using {file_names} and key: {info_key}")

ds_info_list = []

for f in file_names:
    with open(f, "r") as f:
        data = json.load(f)
        if info_key != "inter_class_std" and info_key not in data:
            print("------------------------------")
            print(f"Could not find key {info_key} in ds-info.json!")
            print("------------------------------")
            sys.exit(1)
        ds_info_list.append(data)

if not is_class_wise:
    print(" --- Plotting Average Metric ---")
    fig, ax = plt.subplots(layout="constrained")
    name_list = []
    value_list = []
    for info in ds_info_list:
        name = info["name"]
        if name == "emnist-medium-balanced":
            name = "emnist"

        if info_key == "inter_class_std":
            value = info["class_std"]
            value = np.array(list(value.values())).std()
        else:
            value = info[info_key]
        name_list.append(name)
        value_list.append(value)

    figure_filename = f"bar_chart_{info_key}_{'_'.join(name_list)}.png"
    ax.bar(name_list, value_list)
    if y_name is None:
        ax.set_ylabel(info_key)
    else:
        ax.set_ylabel(y_name)

    plt.savefig(figure_filename)

else:
    print(" --- Plotting Class Wise Metric ---")
    fig, ax = plt.subplots(layout="constrained")

    value_dict: Dict[str, List[float]] = defaultdict(list)

    for info in ds_info_list:
        name = info["name"]
        if name == "emnist-medium-balanced":
            name = "emnist"

        class_value_dict = info[info_key]
        class_value_dict = dict(sorted(class_value_dict.items()))
        value_dict[name] = list(class_value_dict.values())

    x = np.arange(len(list(value_dict.values())[0]))
    width = 0.22
    multiplier = 0

    figure_filename = (
        f"class_wise_bar_chart_{info_key}_{'_'.join(value_dict.keys())}.png"
    )

    for ds_name, values in value_dict.items():
        offset = width * multiplier
        ax.bar(x + offset, values, width, label=ds_name)
        multiplier += 1

    if y_name is None:
        ax.set_ylabel(info_key)
    else:
        ax.set_ylabel(y_name)
    ax.set_xlabel("Class")
    ax.set_xticks(x + width, x)
    ax.legend()
    plt.savefig(figure_filename)
