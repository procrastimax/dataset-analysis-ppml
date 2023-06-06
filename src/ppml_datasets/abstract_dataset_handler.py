import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from tensorflow.keras.layers import Layer, Resizing, Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import scipy.stats
import os
import math
from io import BytesIO

import PIL

import sys

from ppml_datasets.utils import get_ds_as_numpy


@dataclass(eq=True, frozen=False)
class AbstractDataset():
    dataset_name: str
    dataset_path: Optional[str]
    # shape that the dataset should be transformed to
    model_img_shape: Tuple[int, int, int]

    batch_size: Optional[int]
    convert_to_rgb: bool
    augment_train: bool

    shuffle: bool
    is_tfds_ds: bool
    num_classes: int

    # if True, automatically builds ds_info after loading dataset data
    builds_ds_info: bool = field(default=False, repr=False)

    # model specific preprocessing for the dataset like: tf.keras.applications.resnet50.preprocess_input
    preprocessing_function: Optional[Callable[[float], tf.Tensor]] = None

    variants: Optional[List[Dict]] = None

    # image shape of the original dataset data (currently has no real function but interesting to know)
    dataset_img_shape: Optional[Tuple[int, int, int]] = None
    # optionally providable class_names, only for cosmetic purposes when printing out ds_info
    class_names: Optional[List[str]] = None

    random_rotation: Optional[float] = 0.1
    random_zoom: Optional[float] = 0.15
    random_flip: Optional[str] = "horizontal"
    random_brightness: Optional[float] = 0.1
    random_translation_width: Optional[float] = 0.1
    random_translation_height: Optional[float] = 0.1

    random_seed: int = 42
    repeat: bool = False

    class_labels: Optional[Tuple[Any]] = None
    class_counts: Optional[Tuple[int]] = None
    class_distribution: Optional[Tuple[int]] = None

    train_val_test_split: Tuple[float, float, float] = field(init=False)

    ds_info: Dict[str, Any] = field(init=False, default_factory=dict)
    ds_train: tf.data.Dataset = field(init=False, repr=False)
    ds_val: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_test: Optional[tf.data.Dataset] = field(init=False, repr=False, default=None)

    ds_attack_train: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_attack_test: tf.data.Dataset = field(init=False, repr=False, default=None)

    def _load_dataset(self):
        """Load dataset from tfds library.

        This function should be overwritten by all classes which do not utilize the tfds library to load the dataset.
        Overwrite this function with the needed functionality to load the dataset from files. Then call the 'load_dataset()' function to bundle
        data loading and dataset info creation.
        """
        if self.is_tfds_ds:
            self._load_from_tfds()

    def load_dataset(self):
        print(f"Loading {self.dataset_name}")
        self._load_dataset()
        if self.builds_ds_info:
            self.build_ds_info()

    def _load_from_tfds(self):
        """Load dataset from tensorflow_datasets via 'dataset_name'."""
        if not self.is_tfds_ds:
            print("Cannot load dataset from tfds since it is not a tfds dataset!")
            return

        if self.dataset_path is not None:
            data_dir = os.path.join(self.dataset_path, self.dataset_name)
        else:
            data_dir = None

        ds_dict: dict = tfds.load(
            name=self.dataset_name,
            data_dir=data_dir,
            as_supervised=True,
            with_info=False
        )

        if "val" in ds_dict.keys():
            self.ds_val = ds_dict["val"]
            print("Loaded validation DS")
        if "test" in ds_dict.keys():
            self.ds_test = ds_dict["test"]
            print("Loaded test DS")
        if "train" in ds_dict.keys():
            self.ds_train = ds_dict["train"]
            print("Loaded train DS")

    def split_val_from_train(self, val_split: float = 0.3) -> Tuple[int, int]:
        """Split train dataset into validation and train dataset.

        Returns new length of train and validation DS
        """
        self.ds_train, self.ds_val = tf.keras.utils.split_dataset(
            self.ds_train, right_size=val_split,
            shuffle=True, seed=self.random_seed)

        return (len(self.ds_train), len(self.ds_val))

    def resplit_datasets(self, train_val_test_split: Tuple[float, float, float], percentage_loaded_data: int = 100):
        """Resplits all datasets (train, val, test) into new split values.

        First all current datasets are merged into one, than the datasets are resplitted into the specified split parts.
        If percentage_loaded_data is specified, than only this fraction of the merged dataset is used for splitting,
        effectively reducing the number of samples in each dataset.
        """
        ds = self.ds_train
        if self.ds_test is not None:
            ds = ds.concatenate(self.ds_test)

        if self.ds_val is not None:
            ds = ds.concatenate(self.ds_val)

        if percentage_loaded_data != 100:
            new_ds_size = math.ceil(len(ds) * (self.percentage_loaded_data / 100.0))
            ds = ds.take(new_ds_size)

        train_split = self.train_val_test_split[0]
        val_split = self.train_val_test_split[1]
        test_split = self.train_val_test_split[2]

        self.ds_train, right_ds = tf.keras.utils.split_dataset(
            ds, left_size=train_split,
            shuffle=True, seed=self.random_seed)

        if val_split == 0.0:
            self.ds_test = right_ds
        elif test_split == 0.0:
            self.ds_val = right_ds
        else:
            # shuffling once should be enough
            self.ds_val, self.ds_test = tf.keras.utils.split_dataset(
                right_ds, left_size=val_split / (val_split + test_split), shuffle=False)

    def reduce_samples_per_class_train_ds(self, max_samples_per_class: int):
        sample_counters = defaultdict(int)
        reduced_samples = []

        for sample, label in self.ds_train:
            if sample_counters[label.numpy().tolist()] < max_samples_per_class:
                sample_counters[label.numpy().tolist()] += 1
                reduced_samples.append((sample, label))

        reduced_dataset = tf.data.Dataset.from_generator(
            lambda: (sample_label for sample_label in reduced_samples),
            output_signature=(
                tf.TensorSpec(shape=sample.shape, dtype=sample.dtype),
                tf.TensorSpec(shape=label.shape, dtype=label.dtype)
            )
        )
        self.ds_train = reduced_dataset
        ds_len = sum(1 for _ in self.ds_train)
        self.ds_train = self.ds_train.apply(tf.data.experimental.assert_cardinality(ds_len))
        print(f"Reduced class size to {max_samples_per_class}")

    def merge_all_datasets(self, percentage_loaded_data: int = 100) -> tf.data.Dataset:
        """Merge all datasets (train, val, test) into train dataset.

        A percentage can be specified, than only this percentage of the old data is used for the new train_ds after merging.
        """
        ds = self.ds_train
        if self.ds_test is not None:
            ds = ds.concatenate(self.ds_test)

        if self.ds_val is not None:
            ds = ds.concatenate(self.ds_val)

        if percentage_loaded_data != 100:
            new_ds_size = math.ceil(len(ds) * (self.percentage_loaded_data / 100.0))
            ds = ds.take(new_ds_size)

        self.ds_train = ds

    def set_class_names(self, class_names: List[str]):
        self.class_names = class_names

    def prepare_datasets(self):
        """Prepare all currently stored datasets (train, val, test) and the corresponding attack datsets (train, test).

        Preparation can include data shuffling, augmentation and resnet50-preprocessing.
        Augmentation is applied to train dataset if specified, augmentation is never applied to validation or test dataset

        """
        # prepare attack datasets
        # we need to first prepare the attack DS since they depend on the unmodified original datasets
        self.ds_attack_train = self.prepare_ds(self.ds_train, cache=True, resize_rescale=True,
                                               img_shape=self.model_img_shape,
                                               batch_size=1, convert_to_rgb=self.convert_to_rgb,
                                               preprocessing_func=self.preprocessing_function,
                                               shuffle=False, augment=False)
        if self.ds_test is not None:
            self.ds_attack_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True,
                                                  img_shape=self.model_img_shape,
                                                  batch_size=1, convert_to_rgb=self.convert_to_rgb,
                                                  preprocessing_func=self.preprocessing_function,
                                                  shuffle=False, augment=False)

        self.ds_train = self.prepare_ds(self.ds_train, cache=True, resize_rescale=True,
                                        img_shape=self.model_img_shape,
                                        batch_size=self.batch_size, convert_to_rgb=self.convert_to_rgb,
                                        preprocessing_func=self.preprocessing_function,
                                        shuffle=self.shuffle, augment=self.augment_train)

        if self.ds_val is not None:
            self.ds_val = self.prepare_ds(self.ds_val, cache=True, resize_rescale=True,
                                          img_shape=self.model_img_shape,
                                          batch_size=self.batch_size,
                                          convert_to_rgb=self.convert_to_rgb,
                                          preprocessing_func=self.preprocessing_function,
                                          shuffle=False, augment=False)

        if self.ds_test is not None:
            self.ds_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True,
                                           img_shape=self.model_img_shape, batch_size=self.batch_size,
                                           convert_to_rgb=self.convert_to_rgb,
                                           preprocessing_func=self.preprocessing_function,
                                           shuffle=False, augment=False)

    def prepare_ds(self, ds: tf.data.Dataset,
                   resize_rescale: bool,
                   img_shape: Tuple[int, int, int],
                   batch_size: Optional[int],
                   convert_to_rgb: bool,
                   preprocessing_func: Optional[Callable[[float], tf.Tensor]],
                   shuffle: bool,
                   augment: bool,
                   cache: Union[str, bool] = True) -> tf.data.Dataset:
        """Prepare datasets for training and validation for the ResNet50 model.

        This function applies image resizing, resnet50-preprocessing to the dataset. Optionally the data can be shuffled or further get augmented (random flipping, etc.)

        Parameter
        --------
        ds: tf.data.Dataset - dataset used for preparation steps
        resize_rescale: bool - if True, resizes the dataset to 'img_shape' and rescales all pixel values to a value between 0 and 255
        img_shape: Tuple[int, int, int] - if resize_rescale is True, than this value is used to rescale the image data to this size, consist of [height, width, color channel] -> only width and height are used for rescaling
        batch_size: int | None - batch size specified by integer value, if None is passed, no batching is applied to the data
        convert_to_rgb: bool - if True, the data is converted vom grayscale to rgb values
        preprocessing: bool - if True, model specific preprocessing is applied to the data (currently resnet50_preprocessing)
        shuffle: bool - if True, the data is shuffled, the used shuffle buffer for this has the size of the data
        augment: bool - if True, data augmentation (random flip, random rotation, random translation, random zoom, random brightness) is applied to the data

        """
        AUTOTUNE = tf.data.AUTOTUNE

        preprocessing_layers = tf.keras.models.Sequential()
        if convert_to_rgb:
            preprocessing_layers.add(GrayscaleToRgb())

        if resize_rescale:
            preprocessing_layers.add(Resizing(img_shape[0], img_shape[1]))
            preprocessing_layers.add(Rescaling(scale=1. / 255.))

        if preprocessing_func:
            preprocessing_layers.add(ModelPreprocessing(preprocessing_func))

        if convert_to_rgb or resize_rescale or preprocessing_func:
            ds = ds.map(lambda x, y: (preprocessing_layers(x), y),
                        num_parallel_calls=AUTOTUNE)

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(buffer_size=ds.cardinality().numpy(), seed=self.random_seed)

        if batch_size is not None:
            ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)

        if augment:
            augmentation_layers = tf.keras.models.Sequential()

            if self.random_flip:
                augmentation_layers.add(RandomFlip(self.random_flip))

            if self.random_rotation:
                augmentation_layers.add(RandomRotation(self.random_rotation, fill_mode="constant"))

            if self.random_translation_width and self.random_translation_height:
                augmentation_layers.add(RandomTranslation(self.random_translation_height,
                                                          self.random_translation_width, fill_mode="constant"))
            if self.random_zoom:
                augmentation_layers.add(RandomZoom(self.random_zoom, fill_mode="constant"))

            if self.random_brightness:
                augmentation_layers.add(RandomBrightness(self.random_brightness))

            ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    def calculate_class_weights(self) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, float]]]:
        """Calculate class weights and class counts for train dataset."""
        class_labels, class_counts, class_distribution = self.get_class_distribution()

        class_counts_dict: Dict[str, int] = {}
        for y, count in zip(class_labels, class_counts):
            if self.class_names is not None and len(self.class_names) == len(class_labels):
                class_counts_dict[f"{self.class_names[y]}({y})"] = count
            else:
                class_counts_dict[y] = count

        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(class_distribution),
                                                    y=class_distribution)

        class_weights: Dict[str, float] = {}
        if self.class_names is not None and len(self.class_names) == len(class_labels):
            for i, weight in enumerate(weights):
                class_weights[f"{self.class_names[y]}({y})"] = weight
        else:
            class_weights = dict(enumerate(weights))
        return (class_counts_dict, class_weights)

    def get_data_histogram(self, use_mean: bool = False) -> Tuple[np.array, np.array]:
        """Calculate histogram from train datasets.

        Return:
        ------
        Tuple[np.array, np.narray] -> (hist, bins)

        """
        samples = np.array([sample for (sample, _) in list(tfds.as_numpy(self.ds_train))])

        if use_mean:
            samples = np.mean(samples, axis=(0))

        hist, bins = np.histogram(samples, bins=range(255), density=True)
        return (hist, bins)

    def get_dataset_count(self) -> Dict[str, int]:
        """Calculate number of datapoints for each part of the dataset (train,test,val)."""
        ds_count: Dict[str, int] = defaultdict(int)
        if self.ds_train is not None:
            ds_count["train"] = self.ds_train.cardinality().numpy()

        if self.ds_val is not None:
            ds_count["val"] = self.ds_val.cardinality().numpy()

        if self.ds_test is not None:
            ds_count["test"] = self.ds_test.cardinality().numpy()

        return ds_count

    def get_class_distribution(self, ds: Optional[tf.data.Dataset] = None, force_recalcuation: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and return absolute class distribution from train dataset.

        This function returns the desired class_labels, class_counts and class_distribution values but also sets these variables as class variables.
        This is useful to not execute the function again, but only return the class variables, unless the 'force_recalcuation' flag is set to True.

        Parameter:
        --------
        ds: tf.data.Dataset - an optional dataset can be given to this function to calculate the class distribution of the given datasets
                              if ds is not set, it is assumed to calculate the class distribution from the current train dataset
        force_recalcuation: bool - (default = False), if set to True, this function calculates the class_distribution again

        Return:
        ------
        (np.ndarray, np.ndarray, np.ndarray): three numpy arrays
            -> first one containing the class number
            -> second one containing the number of datapoints in the class (ordered)
            -> third one as a class representation for all datapoints
        f.e.: ([1,2,3,4,5],[404,133,313,122,10], [4,1,0,2,5,4,1,4,3,2,4,3,3,1,...])

        """
        if self.class_counts is not None and self.class_labels is not None and self.class_distribution is not None and force_recalcuation is not True:
            return (self.class_labels, self.class_counts, self.class_distribution)

        if ds is not None:
            y_train = np.fromiter(ds.map(lambda _, y: y), int)
        else:
            y_train = np.fromiter(self.ds_train.map(lambda _, y: y), int)

        distribution = np.unique(y_train, return_counts=True)

        self.class_labels = distribution[0]
        self.class_counts = distribution[1]
        self.class_distribution = y_train

        return distribution + (y_train,)

    def calculate_class_imbalance(self) -> float:
        """Calculate class imbalance value for the train dataset.

        Idea of using shannon entropy to calculate balance from here: https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        For data of n instances, if we have k classes of size c_i, we can calculate the entropy

        Return:
        ------
        float:  class imbalance value [0,1]
                0 - unbalanced dataset
                1 - balanced dataset

        """
        _, class_counts, _ = self.get_class_distribution()

        n: int = sum(class_counts)
        k: int = len(class_counts)
        H: float = 0.0
        for c in class_counts:
            H += (c / n) * np.log((c / n))

        H *= -1
        B: float = H / np.log(k)
        return B

    def calculate_compressed_image_size(self) -> Dict[int, np.array]:
        """Calculate compressed image size of all train dataset images.

        This function needs to be called before preprocessing the dataset.

        Returns a dict of numpy array mapped to classes.
        The numpy array contain for each image the following values in this order: entropy, uncompressed_size, png_size, png_ratio, jpeg_size, jpeg_ratio

        """
        class_dict: Dict[int, np.array] = {}

        for (img, label) in self.ds_train:
            label = label.numpy().astype("uint8")
            # check if grayscale or color image
            if img.shape[2] == 1:
                compressed_img: PIL.Image.Image = PIL.Image.fromarray(
                    img[:, :, 0].numpy().astype("uint8"))
            else:
                compressed_img: PIL.Image.Image = PIL.Image.fromarray(img.numpy().astype("uint8"))

            entropy = compressed_img.entropy()

            uncompressed_size = len(compressed_img.tobytes())

            buffer = BytesIO()
            compressed_img.save(buffer, format="PNG", optimize=True)
            png_size = buffer.tell()
            png_ratio = png_size / uncompressed_size

            buffer = BytesIO()
            compressed_img.save(buffer, format="JPEG", optimize=True, quality=75)
            jpeg_size = buffer.tell()
            jpeg_ratio = jpeg_size / uncompressed_size

            values = np.array([entropy, uncompressed_size, png_size,
                               png_ratio, jpeg_size, jpeg_ratio])

            if label not in class_dict:
                class_dict[label] = values
            else:
                current = class_dict[label]
                class_dict[label] = np.vstack((current, values))

        # convert keys from uint8 to int for jsonify
        class_dict = {int(k): v for k, v in class_dict.items()}
        return class_dict

    def calculate_image_fractal_dimension(self) -> Dict[int, List[float]]:
        """Calculate the fractal dimension of all images."""
        print("Calculating fractal dimension of all images")

        def grayConversion(image):
            grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
            gray_img = grayValue.astype(np.uint8)
            return gray_img

        counter = 0
        class_dict: Dict[int, List[float]] = defaultdict(list)
        for (img, label) in self.ds_train:
            label = int(label.numpy().astype("uint8"))

            # check if grayscale or color image, convert to grayscale if RGB
            if img.shape[2] == 1:
                img = img.numpy().astype("uint8")
            else:
                img = grayConversion(img.numpy().astype("uint8"))

            img = img[:, :, 0]
            fractal_dim = self._fractal_dimension(img)
            class_dict[label].append(fractal_dim)
            counter += 1
            if counter % 100 == 0:
                print(f"Calculated fractal dimension of {counter} images")

        return class_dict

    def _fractal_dimension(self, image: np.ndarray) -> np.float64:
        """Calculate the fractal dimension of an image represented by a 2D numpy array.

        Code used from: https://github.com/brian-xu/FractalDimension/blob/master/FractalDimension.py
        The algorithm is a modified box-counting algorithm as described by Wen-Li Lee and Kai-Sheng Hsieh.

        Args:
        ----
        image: A 2D array containing a grayscale image. Format should be equivalent to cv2.imread(flags=0).
                   The size of the image has no constraints, but it needs to be square (m×m array).

        Return:
        ------
        D: The fractal dimension Df, as estimated by the modified box-counting algorithm.

        """
        M = image.shape[0]  # image shape
        G_min = image.min()  # lowest gray level (0=white)
        G_max = image.max()  # highest gray level (255=black)
        G = G_max - G_min + 1  # number of gray levels, typically 256
        prev = -1  # used to check for plateaus
        r_Nr = []

        for L in range(2, (M // 2) + 1):
            h = max(1, G // (M // L))  # minimum box height is 1
            N_r = 0
            r = L / M
            for i in range(0, M, L):
                # create enough boxes with height h to fill the fractal space
                boxes = [[]] * ((G + h - 1) // h)
                for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                    for pixel in row[i:i + L]:
                        # lowest box is at G_min and each is h gray levels tall
                        height = (pixel - G_min) // h
                        boxes[height].append(pixel)  # assign the pixel intensity to the correct box
                # calculate the standard deviation of each box
                stddev = np.sqrt(np.var(boxes, axis=1))
                # remove boxes with NaN standard deviations (empty)
                stddev = stddev[~np.isnan(stddev)]
                nBox_r = 2 * (stddev // h) + 1
                N_r += sum(nBox_r)
            if N_r != prev:  # check for plateauing
                r_Nr.append([r, N_r])
                prev = N_r
        x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
        y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
        D = np.polyfit(x, y, 1)[0]  # D = lim r -> 0 log(Nr)/log(1/r)
        return D

    def build_ds_info(self):
        """Build dataset info dictionary.

        This function needs to be called after initializing and loading the dataset, but before calling preprocessing on it!

        """
        fractal_dim_dict = self.calculate_image_fractal_dimension()
        avg_class_fractal_dim: Dict[int, float] = {}
        avg_ds_fractal_dim: float = 0.0
        for k, v in fractal_dim_dict.items():
            avg_ds_fractal_dim += np.sum(v)
            avg_class_fractal_dim[k] = np.mean(v)
        avg_ds_fractal_dim = avg_ds_fractal_dim / len(fractal_dim_dict.items())

        compression_dict = self.calculate_compressed_image_size()
        # calculate metrics for every class and for the whole DS
        dataset_compression = None
        for val in compression_dict.values():
            if dataset_compression is None:
                dataset_compression = val
            else:
                dataset_compression = np.vstack((dataset_compression, val))

        avg_ds_entropy = np.average(dataset_compression[:, 0])
        avg_ds_byte_size = np.average(dataset_compression[:, 1])
        avg_ds_png_size = np.average(dataset_compression[:, 2])
        avg_ds_png_ratio = np.average(dataset_compression[:, 3])
        avg_ds_jpeg_size = np.average(dataset_compression[:, 4])
        avg_ds_jpeg_ratio = np.average(dataset_compression[:, 5])

        avg_class_entropy: Dict[int, float] = {}
        avg_class_png_size: Dict[int, float] = {}
        avg_class_png_ratio: Dict[int, float] = {}
        avg_class_jpeg_size: Dict[int, float] = {}
        avg_class_jpeg_ratio: Dict[int, float] = {}

        for k, v in compression_dict.items():
            avg_entropy = np.average(v[:, 0])
            avg_png_size = np.average(v[:, 2])
            avg_png_ratio = np.average(v[:, 3])
            avg_jpeg_size = np.average(v[:, 4])
            avg_jpeg_ratio = np.average(v[:, 5])

            avg_class_entropy[k] = avg_entropy
            avg_class_png_size[k] = avg_png_size
            avg_class_png_ratio[k] = avg_png_ratio
            avg_class_jpeg_size[k] = avg_jpeg_size
            avg_class_jpeg_ratio[k] = avg_jpeg_ratio

        class_counts, class_weights = self.calculate_class_weights()
        ds_count = self.get_dataset_count()
        total_count: int = sum(ds_count.values())
        class_imbalance: float = self.calculate_class_imbalance()

        # convert int64 keys to int keys -> to jsonify
        class_counts = {str(k): int(v) for k, v in class_counts.items()}
        class_weights = {str(k): int(v) for k, v in class_weights.items()}

        self.ds_info = {
            'name': self.dataset_name,  # not useful for dataframe
            'dataset_img_shape': self.dataset_img_shape,
            'model_img_shape': self.model_img_shape,
            'total_count': total_count,
            'train_count': ds_count["train"],
            'val_count': ds_count["val"],
            'test_count': ds_count["test"],
            'num_classes': self.num_classes,
            'class_imbalance': class_imbalance,
            'avg_byte_count': avg_ds_byte_size,
            'avg_entropy': avg_ds_entropy,
            'avg_png_size': avg_ds_png_size,
            'avg_png_ratio': avg_ds_png_ratio,
            'avg_jpeg_size': avg_ds_jpeg_size,
            'avg_jpeg_ratio': avg_ds_jpeg_ratio,
            'avg_fractal_dim': avg_ds_fractal_dim,

            'class_weights': class_weights,  # not useful for dataframe
            'class_counts': class_counts,  # not useful for dataframe
            'class_avg_entropy': avg_class_entropy,  # not useful for dataframe
            'class_avg_png_size': avg_class_png_size,  # not useful for dataframe
            'class_avg_png_ratio': avg_class_png_ratio,  # not useful for dataframe
            'class_avg_jpeg_size': avg_class_jpeg_size,  # not useful for dataframe
            'class_avg_jpeg_ratio': avg_class_jpeg_ratio,  # not useful for dataframe
            'class_avg_fractal_dim': avg_class_fractal_dim,  # not useful for dataframe
        }

    def get_ds_info_as_df(self) -> pd.DataFrame:

        # modify dict to make suitable dataframe from it
        dict_cpy = self.ds_info.copy()
        del (dict_cpy["class_counts"])
        del (dict_cpy["class_weights"])
        del (dict_cpy["class_avg_entropy"])
        del (dict_cpy["class_avg_png_size"])
        del (dict_cpy["class_avg_png_ratio"])
        del (dict_cpy["class_avg_jpeg_size"])
        del (dict_cpy["class_avg_jpeg_ratio"])
        del (dict_cpy["class_avg_fractal_dim"])
        del (dict_cpy["name"])

        dict_cpy["dataset_img_shape"] = "/".join(map(str, dict_cpy["dataset_img_shape"]))
        dict_cpy["model_img_shape"] = "/".join(map(str, dict_cpy["model_img_shape"]))

        df = pd.DataFrame(dict_cpy, index=[self.dataset_name])

        return df

    def get_train_ds_subset(self, keep: np.ndarray, apply_processing: bool = False) -> tf.data.Dataset:
        """Return only a subset of datapoints from the training dataset.

        The subset is specified by a boolean numpy array.

        Before creating the subset, the dataset gets unbatched.
        If apply_processing is set to True, then all processing steps like resizing, augmentation, etc. are applied.
        If apply_processing is set to False, only caching, batching and prefetching is applied.
        """
        (values, labels) = self.get_train_ds_as_numpy()

        values = values[keep]
        labels = labels[keep]

        ds = tf.data.Dataset.from_tensor_slices((values, labels))

        if apply_processing:
            ds = self.prepare_ds(ds, cache=True, resize_rescale=True,
                                 img_shape=self.model_img_shape,
                                 batch_size=self.batch_size, convert_to_rgb=self.convert_to_rgb,
                                 preprocessing_func=self.preprocessing_function,
                                 shuffle=self.shuffle, augment=self.augment_train)
        else:
            ds = ds.cache().batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
                buffer_size=tf.data.AUTOTUNE)

        return ds

    def get_train_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_train)

    def get_test_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_test)

    def get_val_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Validation Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_val)

    def get_attack_train_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_train)

    def get_attack_test_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_test)


class GrayscaleToRgb(Layer):
    """Layer for converting 1-channel grayscale input to 3-channel rgb."""

    def __init__(self, **kwargs):
        """Initialize GrayscaleToRgb layer."""
        super().__init__(**kwargs)

    def call(self, x):
        return tf.image.grayscale_to_rgb(x)


class RgbToGrayscale(Layer):
    """Layer for converting 3-channel rgb input to 1-channel grayscale."""

    def __init__(self, **kwargs):
        """Initialize RgbToGrayscale layer."""
        super().__init__(**kwargs)

    def call(self, x):
        return tf.image.rgb_to_grayscale(x)


class RandomBrightness(Layer):
    """Layer for random brightness augmentation in images."""

    def __init__(self, factor=0.1, **kwargs):
        """Initialize RandomBrightness layer."""
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return tf.image.random_brightness(x, max_delta=self.factor)


class ModelPreprocessing(Layer):
    """Layer for specific model preprocessing steps."""

    def __init__(self, pre_func: Callable[[float], tf.Tensor], **kwargs):
        """Initialize layer for model preprocessing."""
        super().__init__(**kwargs)
        self.pre_func = pre_func

    def call(self, x):
        return self.pre_func(x)
