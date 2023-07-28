from typing import Callable, Optional, Tuple

import tensorflow as tf

from ppml_datasets.abstract_dataset_handler import (
    AbstractDataset,
    AbstractDatasetClassImbalance,
    AbstractDatasetClassSize,
    AbstractDatasetCustomClasses,
)


class EMNISTLargeUnbalancedDataset(AbstractDataset):
    """An unbalanced datset with 62 classes, representing Digits and Letters (upper  and lowercase) and 697,932 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/byclass",
            dataset_name="emnist-large-unbalanced",
            dataset_path=dataset_path,
            dataset_img_shape=(28, 28, 1),
            random_seed=42,
            num_classes=62,
            model_img_shape=model_img_shape,
            batch_size=batch_size,
            convert_to_rgb=True,
            augment_train=augment_train,
            preprocessing_function=preprocessing_func,
            shuffle=True,
            is_tfds_ds=True,
            builds_ds_info=builds_ds_info,
        )


class EMNISTMediumUnbalancedDataset(AbstractDataset):
    """An unbalanced dataset with 47 classes (digits and letters) with 697,932 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/bymerge",
            dataset_name="emnist-medium-unbalanced",
            dataset_path=dataset_path,
            dataset_img_shape=(28, 28, 1),
            random_seed=42,
            num_classes=47,
            model_img_shape=model_img_shape,
            batch_size=batch_size,
            convert_to_rgb=True,
            augment_train=augment_train,
            preprocessing_function=preprocessing_func,
            shuffle=True,
            is_tfds_ds=True,
            builds_ds_info=builds_ds_info,
        )


class EMNISTMediumBalancedDataset(AbstractDataset):
    """An balanced dataset with 47 classes (digits and letters) with 112,800 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/balanced",
            dataset_name="emnist-medium-balanced",
            dataset_path=dataset_path,
            dataset_img_shape=(28, 28, 1),
            random_seed=42,
            num_classes=47,
            model_img_shape=model_img_shape,
            batch_size=batch_size,
            convert_to_rgb=True,
            augment_train=augment_train,
            preprocessing_function=preprocessing_func,
            shuffle=True,
            is_tfds_ds=True,
            builds_ds_info=builds_ds_info,
        )


class EMNISTLettersBalancedDataset(AbstractDataset):
    """An balanced dataset with 37 classes (letters) with 88,800 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/letters",
            dataset_name="emnist-letters-balanced",
            dataset_path=dataset_path,
            dataset_img_shape=(28, 28, 1),
            random_seed=42,
            num_classes=37,
            model_img_shape=model_img_shape,
            batch_size=batch_size,
            convert_to_rgb=True,
            augment_train=augment_train,
            preprocessing_function=preprocessing_func,
            shuffle=True,
            is_tfds_ds=True,
            builds_ds_info=builds_ds_info,
        )


class EMNISTDigitsManyBalancedDataset(AbstractDataset):
    """An balanced dataset with 10 classes (digits) with 88,800 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/digits",
            dataset_name="emnist-digits-balanced",
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
            builds_ds_info=builds_ds_info,
        )


class EMNISTDigitsNormalBalancedDataset(AbstractDataset):
    """An balanced dataset with 10 classes (digits) with 60,000 train samples."""

    def __init__(
        self,
        model_img_shape: Tuple[int, int, int],
        builds_ds_info: bool = False,
        batch_size: int = 32,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
        augment_train: bool = True,
        dataset_path: str = "data",
    ):
        super().__init__(
            tfds_name="emnist/mnist",
            dataset_name="emnist-mnist-balanced",
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
            builds_ds_info=builds_ds_info,
        )
