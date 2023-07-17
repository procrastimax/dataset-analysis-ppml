import tensorflow as tf
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from typing import Tuple, Callable, Optional


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
