import tensorflow as tf
from typing import Callable, Tuple, Optional

from ppml_datasets.abstract_dataset_handler import AbstractDataset, AbstractDatasetClassSize, AbstractDatasetClassImbalance


class SVHNDataset(AbstractDataset):
    def __init__(self,
                 model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="svhn_cropped",
                         dataset_name="svhn",
                         dataset_path=dataset_path,
                         dataset_img_shape=(32, 32, 3),
                         random_seed=42,
                         num_classes=10,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class SVHNDatasetClassSize(AbstractDatasetClassSize):
    def __init__(self,
                 ds: SVHNDataset,
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


class SVHNDatasetClassImbalance(AbstractDatasetClassImbalance):
    def __init__(self,
                 ds: SVHNDataset,
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
