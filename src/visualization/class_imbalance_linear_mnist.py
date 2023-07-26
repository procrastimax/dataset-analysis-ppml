import matplotlib.pyplot as plt

from ppml_datasets import MnistDataset, MnistDatasetClassImbalance

model_input_shape = [32, 32, 3]
random_seed: int = 42

mnist = MnistDataset(model_input_shape, builds_ds_info=False, augment_train=False)
mnist.load_dataset()
classes, class_counts, _ = mnist.get_class_distribution()
zipped = zip(classes, class_counts)
zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
classes, class_counts = zip(*zipped)
classes = list(map(str, classes))

mnist_1 = MnistDatasetClassImbalance(mnist, imbalance_mode="L", imbalance_ratio=0.3)
mnist_1.load_dataset()
classes_1, class_counts_1, _ = mnist_1.get_class_distribution()
zipped_1 = zip(classes_1, class_counts_1)
zipped_1 = sorted(zipped_1, key=lambda x: x[1], reverse=True)
classes_1, class_counts_1 = zip(*zipped_1)
classes_1 = list(map(str, classes_1))

mnist_2 = MnistDatasetClassImbalance(mnist, imbalance_mode="L", imbalance_ratio=0.6)
mnist_2.load_dataset()
classes_2, class_counts_2, _ = mnist_2.get_class_distribution()
zipped_2 = zip(classes_2, class_counts_2)
zipped_2 = sorted(zipped_2, key=lambda x: x[1], reverse=True)
classes_2, class_counts_2 = zip(*zipped_2)
classes_2 = list(map(str, classes_2))

fig, axs = plt.subplots(3)
fig.suptitle("Dataset Class Imbalance Comparison")

axs[0].bar(classes, class_counts)
axs[1].bar(classes_1, class_counts_1)
axs[2].bar(classes_2, class_counts_2)

axs[0].set_ylabel("Number of samples per class")
axs[0].set_xlabel("Class Label")
axs[1].set_ylabel("Number of samples per class")
axs[1].set_xlabel("Class Label")
axs[2].set_ylabel("Number of samples per class")
axs[2].set_xlabel("Class Label")

axs[0].set_title("MNIST Imbalanced Linear Mode (i=0)")
axs[1].set_title("MNIST Imbalanced Linear Mode (i=0.3)")
axs[2].set_title("MNIST Imbalanced Linear Mode (i=0.6)")

plt.show()
