from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf

from ppml_datasets.abstract_dataset_handler import (
    AbstractDataset,
    AbstractDatasetClassImbalance,
    AbstractDatasetClassSize,
    AbstractDatasetCustomClasses
)


class FashionMnistDataset(AbstractDataset):

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
            tfds_name="fashion_mnist",
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
            builds_ds_info=builds_ds_info,
        )


class FashionMnistDatasetClassSize(AbstractDatasetClassSize):

    def __init__(self, ds: FashionMnistDataset, class_size: int):
        self.class_size = class_size
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(
            tfds_name=ds.tfds_name,
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
            builds_ds_info=ds.builds_ds_info,
        )


class FashionMnistDatasetClassImbalance(AbstractDatasetClassImbalance):

    def __init__(self, ds: FashionMnistDataset, imbalance_mode: str,
                 imbalance_ratio: float):
        self.imbalance_mode = imbalance_mode
        self.imbalance_ratio = imbalance_ratio
        self.ds_train = ds.ds_train
        self.ds_test = ds.ds_test
        self.ds_val = ds.ds_val
        super().__init__(
            tfds_name=ds.tfds_name,
            num_classes=ds.num_classes,
            dataset_name=
            f"{ds.dataset_name}_i{self.imbalance_mode}{self.imbalance_ratio}",
            dataset_path=ds.dataset_path,
            model_img_shape=ds.model_img_shape,
            dataset_img_shape=ds.dataset_img_shape,
            batch_size=ds.batch_size,
            convert_to_rgb=ds.convert_to_rgb,
            augment_train=ds.augment_train,
            shuffle=ds.shuffle,
            is_tfds_ds=ds.is_tfds_ds,
            builds_ds_info=ds.builds_ds_info,
        )


class FashionMnistDatasetCustomClasses(AbstractDatasetCustomClasses):

    def __init__(self, ds: FashionMnistDataset, new_num_classes: int):
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


def build_fmnist(
    model_input_shape: Tuple[int, int, int],
    batch_size: int,
    mods: Dict[str, List[Any]],
    augment_train: bool = False,
    builds_ds_info: bool = False,
) -> AbstractDataset:
    ds = FashionMnistDataset(
        model_img_shape=model_input_shape,
        builds_ds_info=False,
        batch_size=batch_size,
        augment_train=False,
    )
    ds.load_dataset()

    if "n" in mods:
        num_new_classes = mods["n"][0]
        ds = FashionMnistDatasetCustomClasses(ds, num_new_classes)
        ds.load_dataset()

    if "c" in mods:
        class_size = mods["c"][0]
        ds = FashionMnistDatasetClassSize(ds=ds, class_size=class_size)
        ds.load_dataset()

    if "i" in mods:
        (imbalance_mode, imbalance_ratio) = mods["i"]
        ds = FashionMnistDatasetClassImbalance(ds=ds,
                                               imbalance_mode=imbalance_mode,
                                               imbalance_ratio=imbalance_ratio)
        ds.load_dataset()

    return ds
