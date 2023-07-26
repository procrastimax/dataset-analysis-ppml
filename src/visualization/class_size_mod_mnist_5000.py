import matplotlib.pyplot as plt

from ppml_datasets import MnistDataset, MnistDatasetClassSize

model_input_shape = [32, 32, 3]
random_seed: int = 42

mnist = MnistDataset(model_input_shape, builds_ds_info=False, augment_train=False)
mnist.load_dataset()
classes, class_counts, _ = mnist.get_class_distribution()
zipped = zip(classes, class_counts)
zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
classes, class_counts = zip(*zipped)
classes = list(map(str, classes))

mnist_1 = MnistDatasetClassSize(mnist, class_size=5000)
mnist_1.load_dataset()
classes_1, class_counts_1, _ = mnist_1.get_class_distribution()
zipped_1 = zip(classes_1, class_counts_1)
zipped_1 = sorted(zipped_1, key=lambda x: x[1], reverse=True)
classes_1, class_counts_1 = zip(*zipped_1)
classes_1 = list(map(str, classes_1))

fig, axs = plt.subplots(2)
axs[0].bar(classes, class_counts)
axs[1].bar(classes_1, class_counts_1)

axs[0].set_ylabel("Number of samples per class")
axs[0].set_xlabel("Class Label")
axs[1].set_ylabel("Number of samples per class")
axs[1].set_xlabel("Class Label")

axs[0].set_title("MNIST Imbalanced Linear Mode (i=0)")
axs[1].set_title("MNIST Imbalanced Linear Mode (i=0.3)")

plt.show()
