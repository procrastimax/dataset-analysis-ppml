"""
    This script lists all run-names and runs and their status (completed or not uncompleted).
"""

import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-f",
                    "--filedir",
                    type=str,
                    help="The base filedir to start searching for runs.")
args = parser.parse_args()

base_folder = args.filedir

private_dict_completed = defaultdict(list)
non_private_dict_completed = defaultdict(list)

private_dict_missing = defaultdict(list)
non_private_dict_missing = defaultdict(list)

for root, dirs, files in os.walk(args.filedir):
    if "parameter.json" in files or "parameter.csv" in files:
        run_path, run_num = os.path.split(root)
        if run_num == "combined-runs":
            continue
        base_path, run_name = os.path.split(run_path)

        _, run_mode = os.path.split(base_path)
        if run_mode == "private_cnn":
            private_dict_completed[run_name].append(int(run_num))
        elif run_mode == "cnn":
            non_private_dict_completed[run_name].append(int(run_num))

    # check if same depth as expected "parameter.json" file, and if not present, this run is missing
    else:
        if len(root.split("/")) == 4:
            run_path, run_num = os.path.split(root)
            if run_num == "combined-runs":
                continue
            base_path, run_name = os.path.split(run_path)

            _, run_mode = os.path.split(base_path)
            if run_mode == "private_cnn":
                private_dict_missing[run_name].append(int(run_num))
            elif run_mode == "cnn":
                non_private_dict_missing[run_name].append(int(run_num))

# pretty print dicts
print("Completed Runs:")
print("  Non Private Runs:")
for k, v in non_private_dict_completed.items():
    print(f"    - {k} -> {sorted(v)}")

print("  Private Runs:")
for k, v in private_dict_completed.items():
    print(f"    - {k} -> {sorted(v)}")

print()
print("Missing Runs:")
print("  Non Private Runs:")
for k, v in non_private_dict_missing.items():
    print(f"    - {k} -> {sorted(v)}")

print("  Private Runs:")
for k, v in private_dict_missing.items():
    print(f"    - {k} -> {sorted(v)}")
