import json
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]
print(f"Loading {filename}")

class_sizes = []
class_names = []

with open(filename) as f:
    classes = json.load(f)
    for k, v in classes.items():
        class_sizes.append(v)
        class_names.append(int(k))

filename = filename.split("/")[-1]

plt.rcParams['figure.constrained_layout.use'] = True
plt.figure(figsize=(5, 3))
plt.bar(range(len(class_sizes)), class_sizes, color="royalblue", alpha=0.9)
plt.xticks(class_names)
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.savefig(f"bar_plot_{filename}.png")
