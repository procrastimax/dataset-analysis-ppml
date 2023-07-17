import tensorflow as tf
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from typing import Tuple, Callable, Optional


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
