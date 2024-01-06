"""Simple script to visualize the distribution of samples in a dataset.

Expects the ds_info.json file with the "class_counts" field.
"""

import json
import sys

import matplotlib.pyplot as plt

filename = sys.argv[1]
print(f"Loading {filename}")

class_sizes = []
class_names = []

ds_full_name = "dataset"

with open(filename) as f:
    ds_info = json.load(f)
    if "class_counts" in ds_info:
        classes = ds_info["class_counts"]
        for k, v in classes.items():
            class_sizes.append(v)
            class_names.append(int(k))

    if "name" in ds_info:
        ds_full_name = ds_info["name"]

filename = filename.split("/")[-1]

plt.rcParams["figure.constrained_layout.use"] = True
plt.figure(figsize=(4, 3))
plt.bar(range(len(class_sizes)), class_sizes, color="royalblue", alpha=0.9)
plt.xticks(class_names)
plt.xlabel("Class")
plt.ylabel("# Samples")
plt.savefig(f"class_sample_bar_{ds_full_name}.png")
