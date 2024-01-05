import matplotlib.pyplot as plt
import numpy as np

learning_rate = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.01]
clipping_norm = [0.05, 0.1, 0.5, 1.0]

# learning rate x clipping_norm
clipping_norm_sweep = np.array([
    [0.6078, 0.7411, 0.7388, 0.7089, 0.2682, 0.1605],
    [0.6059, 0.7437, 0.7354, 0.7009, 0.3310, 0.1963],
    [0.6026, 0.7447, 0.7369, 0.7051, 0.2751, 0.1178],
    [0.6127, 0.7441, 0.7471, 0.6933, 0.2429, 0.1499],
])

fig, ax = plt.subplots()
im = ax.imshow(clipping_norm_sweep)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(learning_rate)), labels=learning_rate)
ax.set_yticks(np.arange(len(clipping_norm)), labels=clipping_norm)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(learning_rate)):
    for j in range(len(clipping_norm)):
        text = ax.text(i,
                       j,
                       clipping_norm_sweep[j, i],
                       ha="center",
                       va="center")
plt.ylabel("Clipping Norm")
plt.xlabel("Learning Rate")
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
cbar.ax.set_ylabel("F1-score", rotation=-90, va="bottom")
fig.tight_layout()
plt.savefig("clipping-norm-sweep-heatmap.png")
