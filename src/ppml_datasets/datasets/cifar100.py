import tensorflow as tf
from ppml_datasets.abstract_dataset_handler import AbstractDataset, AbstractDatasetCustomClasses
from typing import Tuple, Callable, Optional


class Cifar100Dataset(AbstractDataset):
    def __init__(self, model_img_shape: Tuple[int, int, int],
                 builds_ds_info: bool = False,
                 batch_size: int = 32,
                 preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None,
                 augment_train: bool = True,
                 dataset_path: str = "data"):
        super().__init__(tfds_name="cifar100",
                         dataset_name="cifar100",
                         dataset_path=dataset_path,
                         random_seed=42,
                         dataset_img_shape=(32, 32, 3),
                         num_classes=100,
                         model_img_shape=model_img_shape,
                         batch_size=batch_size,
                         convert_to_rgb=False,
                         augment_train=augment_train,
                         preprocessing_function=preprocessing_func,
                         shuffle=True,
                         is_tfds_ds=True,
                         builds_ds_info=builds_ds_info)


class Cifar100DatasetCustomClasses(AbstractDatasetCustomClasses):
    def __init__(self,
                 ds: Cifar100Dataset,
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
