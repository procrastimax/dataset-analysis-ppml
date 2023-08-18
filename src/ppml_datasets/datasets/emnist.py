from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf

from ppml_datasets.abstract_dataset_handler import (
    AbstractDataset,
    AbstractDatasetClassImbalance,
    AbstractDatasetClassSize,
    AbstractDatasetCustomClasses,
)


class EMNISTLargeUnbalancedDataset(AbstractDataset):
    """An unbalanced datset with 62 classes, representing Digits and Letters (upper  and lowercase) and 697,932 train samples. [emnist/byclass]"""

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
    """An unbalanced dataset with 47 classes (digits and letters) with 697,932 train samples. [emnist/bymerge]"""

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
    """An balanced dataset with 47 classes (digits and letters) with 112,800 train samples. All classes have a size of 2400. [emnist/balanced]"""

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
    """An balanced dataset with 37 classes (letters) with 88,800 train samples. All classes have a size of ca. 3300/3400. [emnist/letters]"""

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
    """An balanced dataset with 10 classes (digits) with 88,800 train samples. [emnist/digits]"""

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


class EMNISTMediumBalancedDatasetClassSize(AbstractDatasetClassSize):

    def __init__(self, ds: EMNISTMediumBalancedDataset, class_size: int):
        self.class_size = class_size
        self.ds_train = ds.ds_train
        self.ds_val = ds.ds_val
        self.ds_test = ds.ds_test
        super().__init__(
            tfds_name=ds.tfds_name,
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
            builds_ds_info=ds.builds_ds_info,
        )


class EMNISTMediumBalancedDatasetCustomClasses(AbstractDatasetCustomClasses):

    def __init__(self, ds: EMNISTMediumBalancedDataset, new_num_classes: int):
        self.new_num_classes = new_num_classes
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(
            tfds_name=ds.tfds_name,
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
            builds_ds_info=ds.builds_ds_info,
        )


def build_emnist(
    model_input_shape: Tuple[int, int, int],
    batch_size: int,
    mods: Dict[str, List[Any]],
    augment_train: bool = False,
    builds_ds_info: bool = False,
) -> AbstractDataset:
    # ds = EMNISTLargeUnbalancedDataset(
    #    model_img_shape=model_input_shape,
    #    batch_size=batch_size,
    #    augment_train=augment_train,
    #    builds_ds_info=builds_ds_info,
    # )
    # ds = EMNISTMediumUnbalancedDataset(
    #    model_img_shape=model_input_shape,
    #    batch_size=batch_size,
    #    augment_train=augment_train,
    #    builds_ds_info=builds_ds_info,
    # )
    ds = EMNISTMediumBalancedDataset(
        model_img_shape=model_input_shape,
        batch_size=batch_size,
        augment_train=augment_train,
        builds_ds_info=builds_ds_info,
    )
    # ds = EMNISTLettersBalancedDataset(
    #    model_img_shape=model_input_shape,
    #    batch_size=batch_size,
    #    augment_train=augment_train,
    #    builds_ds_info=builds_ds_info,
    # )

    ds.load_dataset()
    _, classes, _ = ds.get_class_distribution()
    print(classes)

    if "n" in mods:
        num_new_classes = mods["n"][0]
        ds = EMNISTMediumBalancedDatasetCustomClasses(
            ds, new_num_classes=num_new_classes)
        ds.load_dataset()

    if "c" in mods:
        class_size = mods["c"][0]
        ds = EMNISTMediumBalancedDatasetClassSize(ds, class_size=class_size)
        ds.load_dataset()

    return ds
