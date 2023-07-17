import gdown
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Callable, Tuple, List, Dict, Optional

from ppml_datasets.abstract_dataset_handler import AbstractDataset, RgbToGrayscale, AbstractDatasetClassSize, AbstractDatasetGray, AbstractDatasetClassImbalance, AbstractDatasetCustomClasses
from ppml_datasets.utils import get_img


class MnistDataset(AbstractDataset):
    def __init__(self,
                 model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="mnist",
                         dataset_name="mnist",
                         dataset_path=dataset_path,
                         dataset_img_shape=(28, 28, 1),
                         random_seed=42,
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=True,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class MnistDatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: MnistDataset,
                 class_size: int):
        self.class_size = class_size
        self.ds_train = ds.ds_train
        self.ds_val = ds.ds_val
        self.ds_test = ds.ds_test
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_img_shape=ds.dataset_img_shape,
                         dataset_name=f"{ds.dataset_name}_c{class_size}",
                         dataset_path=ds.dataset_path,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class MnistDatasetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: MnistDataset,
                 imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
                         dataset_path=ds.dataset_path,
                         dataset_img_shape=ds.dataset_img_shape,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class MnistDatasetCustomClasses(AbstractDatasetCustomClasses):
    def __init__(self,
                 ds: MnistDataset,
                 new_num_classes: int):

        self.new_num_classes = new_num_classes
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=self.new_num_classes,
                         dataset_name=f"{ds.dataset_name}_n{self.new_num_classes}",
                         dataset_path=ds.dataset_path,
                         dataset_img_shape=ds.dataset_img_shape,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class FashionMnistDataset(AbstractDataset):
    def __init__(self,
                 model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="fashion_mnist",
                         dataset_name="fmnist",
                         dataset_path=dataset_path,
                         random_seed=42,
                         dataset_img_shape=(28, 28, 1),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=True,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class FashionMnistDatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: FashionMnistDataset,
                 class_size: int):
        self.class_size = class_size
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_c{class_size}",
                         dataset_path=ds.dataset_path,
                         dataset_img_shape=ds.dataset_img_shape,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class FashionMnistDatasetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: FashionMnistDataset,
                 imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
                         dataset_path=ds.dataset_path,
                         model_img_shape=ds.model_img_shape,
                         dataset_img_shape=ds.dataset_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class Cifar10Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR10 dataset from AbstractDataset class."""
        super().__init__(tfds_name="cifar10",
                         dataset_name="cifar10",
                         dataset_path=dataset_path,
                         random_seed=42,
                         dataset_img_shape=(32, 32, 3),
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class Cifar10DatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: Cifar10Dataset,
                 class_size: int):
        self.class_size = class_size
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_c{class_size}",
                         dataset_path=ds.dataset_path,
                         dataset_img_shape=ds.dataset_img_shape,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class Cifar10DatsetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: Cifar10Dataset,
                 imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
                         dataset_path=ds.dataset_path,
                         model_img_shape=ds.model_img_shape,
                         dataset_img_shape=ds.dataset_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class Cifar10DatasetCustomClasses(AbstractDatasetCustomClasses):
    def __init__(self,
                 ds: Cifar10Dataset,
                 new_num_classes: int):

        self.new_num_classes = new_num_classes
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=self.new_num_classes,
                         dataset_name=f"{ds.dataset_name}_n{self.new_num_classes}",
                         dataset_path=ds.dataset_path,
                         dataset_img_shape=ds.dataset_img_shape,
                         model_img_shape=ds.model_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=ds.convert_to_rgb,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class Cifar10DatsetGray(AbstractDatasetGray):
    def __init__(self,
                 ds: Cifar10Dataset):
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(tfds_name=ds.tfds_name,
                         num_classes=ds.num_classes,
                         dataset_name=f"{ds.dataset_name}_gray",
                         dataset_path=ds.dataset_path,
                         model_img_shape=ds.model_img_shape,
                         dataset_img_shape=ds.dataset_img_shape,
                         batch_size=ds.batch_size,
                         convert_to_rgb=True,
                         augment_train=ds.augment_train,
                         shuffle=ds.shuffle,
                         is_tfds_ds=ds.is_tfds_ds,
                         builds_ds_info=ds.builds_ds_info)


class Cifar100Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        """Initialize the CIFAR100 dataset from AbstractDataset class."""
        super().__init__(dataset_name="cifar100",
                         tfds_name="cifar100",
                         dataset_path=dataset_path,
                         random_seed=42,
                         dataset_img_shape=(32, 32, 3),
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True, is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class ImagenetteDataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None):
        """Initialize the full size v2 imagenette dataset from AbstractDataset class."""
        super().__init__(dataset_name="imagenette/full-size-v2",
                         tfds_name="imagenette/full-size-v2",
                         dataset_path="data",
                         dataset_img_shape=(None, None, 3),
                         random_seed=42,
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         train_val_test_split=(1, 1, 0),
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=True,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class Covid19RadiographyDataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 dataset_path: str,
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None):
        """Initialize the Covid19 dataset from AbstractDataset class."""
        super().__init__(dataset_name="covid19-radiography",
                         dataset_path=dataset_path,
                         dataset_img_shape=(299, 299, 3),
                         num_classes=2,
                         random_seed=42,
                         model_img_shape=model_img_shape,
                         train_val_test_split=(0.8, 0.05, 0.15),
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=True,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=False,
                         builds_ds_info=builds_ds_info)

        variants: List[Dict[str, Optional[str]]] = [
            {'activation': 'relu', 'pretraining': None},
            {'activation': 'relu', 'pretraining': 'imagenet'},
            {'activation': 'relu', 'pretraining': 'pneumonia'},
            {'activation': 'tanh', 'pretraining': None},
            {'activation': 'tanh', 'pretraining': 'imagenet'},
            {'activation': 'tanh', 'pretraining': 'pneumonia'},
        ]
        self.variants = variants

        if self.dataset_path:
            dataset_path = self.dataset_path

        self.base_imgpath: str = os.path.join(dataset_path, "COVID-19_Radiography_Dataset")
        self.normal_imgpath: str = os.path.join(self.base_imgpath, "Normal/images")
        self.covid_imgpath: str = os.path.join(self.base_imgpath, "COVID/images")
        self.imbalance_ratio: float = 1.5

        self.train_attack_data: tf.data.Dataset | None = None
        self.test_attack_data: tf.data.Dataset | None = None

    def _load_dataset(self):
        """Retrieve covid19 dataset.

        Overwrites the _load_dataset functionality of the base class, since we cannot load the dataset from tfds here.
        """
        AUTOTUNE = tf.data.AUTOTUNE

        if not os.path.exists(self.base_imgpath):
            url = 'https://drive.google.com/uc?id=1ZMgUQkwNqvMrZ8QaQmSbiDqXOWAewwou&confirm=t'
            output = os.path.join(self.base_imgpath, 'COVID-19_Radiography_Database.zip')
            gdown.cached_download(url, output, quiet=False, use_cookies=False,
                                  postprocess=gdown.extractall)
            os.remove(output)

        # excluding duplicates normal images from pneumonia pre-training
        excl_imgs = ['Normal-' + str(i) + '.png' for i in range(8852, 10192 + 1)]

        # collect from image paths
        normal_images = [os.path.join(self.normal_imgpath, name) for name in os.listdir(
            self.normal_imgpath) if os.path.isfile(os.path.join(self.normal_imgpath, name)) and name not in excl_imgs]
        covid_images = [os.path.join(self.covid_imgpath, name) for name in os.listdir(
            self.covid_imgpath) if os.path.isfile(os.path.join(self.covid_imgpath, name))]

        # create train-test split
        label_encoding = ['normal', 'COVID-19']  # normal = 0, COVID-19 = 1
        files, labels = [], []

        np.random.shuffle(covid_images)
        files.extend(covid_images)
        labels.extend(np.full(len(covid_images), label_encoding.index('COVID-19')))

        np.random.shuffle(normal_images)
        if self.imbalance_ratio:
            normal_images = normal_images[:int(self.imbalance_ratio * len(covid_images))]
        files.extend(normal_images)
        labels.extend(np.full(len(normal_images), label_encoding.index('normal')))

        files, labels = np.array(files), np.array(labels)

        val_split = self.train_val_test_split[1]
        test_split = self.train_val_test_split[2]
        x_train, x_rest, y_train, y_rest = train_test_split(
            files, labels, test_size=val_split + test_split, random_state=self.random_seed)
        x_test, x_val, y_test, y_val = train_test_split(
            x_rest, y_rest, test_size=val_split / (val_split + test_split), random_state=self.random_seed)

        # build tensorflow dataset
        train_files = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_files = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_files = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # image retrieval
        self.ds_train = train_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_val = val_files.map(get_img, num_parallel_calls=AUTOTUNE)
        self.ds_test = test_files.map(get_img, num_parallel_calls=AUTOTUNE)
