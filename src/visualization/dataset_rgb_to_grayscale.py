from ppml_datasets import Cifar10Dataset, Cifar10DatsetGray
import matplotlib.pyplot as plt

model_input_shape = [32, 32, 3]
random_seed: int = 42

cifar10 = Cifar10Dataset(model_img_shape=model_input_shape,
                         builds_ds_info=False, augment_train=False)
cifar10.load_dataset()

cifar10_gray = Cifar10DatsetGray(ds=cifar10)
cifar10_gray.load_dataset()

x, y = cifar10.get_train_ds_as_numpy(False)
x_g, y_g = cifar10_gray.get_train_ds_as_numpy(False)

fig, ax = plt.subplots(1,2)
ax[0].imshow(x[0].astype('uint8'))
ax[1].imshow(x_g[0].astype('uint8'), cmap='gray')
plt.show()
