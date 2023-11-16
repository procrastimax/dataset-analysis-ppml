import argparse
import os
import sys
import json
import matplotlib.pyplot as plt

FIGSIZE = (5, 5)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--files",
    type=str,
    nargs="+",
    required=True,
    help=
    "List of/ Single Files to be used for creating visualizations. Must be one of the ds-info.json files.",
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
    help=
    "Flag to indicate if the used '-p' parameter specifies a class-wise attribute. The kind of visualization changes if this is the case.",
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


# get current script dir
curr_dir = os.path.abspath(os.getcwd())

for i, f in enumerate(file_names):
    file_names[i] = os.path.join(curr_dir, f)

print(f"Using {file_names} and key: {info_key}")

ds_info_list = []

for f in file_names:
    with open(f, "r") as f:
        data = json.load(f)
        if info_key not in data:
            print(f"Could not find key {info_key} in ds-info.json!")
            sys.exit(1)
        ds_info_list.append(data)

if not is_class_wise:
    fig, ax = plt.subplots()
    name_list = []
    value_list = []
    for info in ds_info_list:
        name = info["name"]
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

