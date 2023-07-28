import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow_datasets as tfds
from imblearn.datasets import make_imbalance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import class_weight
from tensorflow.keras.layers import (
    Layer,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Rescaling,
    Resizing,
)
from tensorflow.keras.utils import to_categorical

from ppml_datasets.piqe import piqe
from ppml_datasets.utils import get_ds_as_numpy, load_dict_from_json, save_dict_as_json


@dataclass(eq=True, frozen=False)
class AbstractDataset:
    tfds_name: Optional[str]
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
    ds_info_path: str = "ds-info"
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
            data_dir = os.path.join(self.dataset_path, self.tfds_name)
        else:
            data_dir = None

        ds_dict: dict = tfds.load(
            name=self.tfds_name, data_dir=data_dir, as_supervised=True, with_info=False
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

    def convert_ds_to_one_hot_encoding(
        self, ds: tf.data.Dataset, unbatch: bool
    ) -> (np.array, np.array):
        (samples, labels) = get_ds_as_numpy(ds, unbatch=unbatch)
        labels = tf.one_hot(labels, self.num_classes)
        labels = np.array(labels, dtype=np.int32)
        return (samples, labels)

    def resplit_datasets(
        self,
        train_val_test_split: Tuple[float, float, float],
        percentage_loaded_data: int = 100,
    ):
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
            ds, left_size=train_split, shuffle=True, seed=self.random_seed
        )

        if val_split == 0.0:
            self.ds_test = right_ds
        elif test_split == 0.0:
            self.ds_val = right_ds
        else:
            # shuffling once should be enough
            self.ds_val, self.ds_test = tf.keras.utils.split_dataset(
                right_ds, left_size=val_split / (val_split + test_split), shuffle=False
            )

    def generate_class_dependent_in_indices(
        self, values: np.array, labels: np.array, reduction_factor: float
    ) -> np.array:
        """Generate a in_indices/keep list dependant on the class size. This reduces the dataset size not generally by a specific factor, but reduces each class by this factor.

        Parameter:
        --------
        values : np.array - samples/ values numpy array
        labels : np.array - labels numpy array
        reduction_factor : float - the factor which specifies the amount of data sampls removed per class

        Return:
        ------
        np.array - keep in_indices

        """

        # check if one-hot encoded f.e. (60000, 10,)
        if len(labels.shape) == 2:
            # convert labels to integer indexing
            labels = tf.argmax(labels, axis=1).numpy()

        class_arrays_in: Dict[int, list] = defaultdict(list)

        for data, label in zip(values, labels):
            class_arrays_in[int(label)].append(data)

        for class_name, data_list in class_arrays_in.items():
            class_arrays_in[class_name] = np.array(data_list)

        class_keep = {}
        for k, v in class_arrays_in.items():
            # generate random bool array with exactly X True values
            keep = np.zeros(len(v), dtype=bool)
            true_indices = np.random.choice(
                keep.size, int(len(v) * reduction_factor), replace=False
            )
            keep.flat[true_indices] = True
            class_keep[k] = keep

        in_indices: List[bool] = []
        for data, label in zip(values, labels):
            in_indices.append(class_keep[label][0])
            class_keep[label] = class_keep[label][1:]

        return np.array(in_indices)

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
        self.ds_attack_train = self.prepare_ds(
            self.ds_train,
            cache=True,
            resize_rescale=True,
            img_shape=self.model_img_shape,
            batch_size=1,
            convert_to_rgb=self.convert_to_rgb,
            preprocessing_func=self.preprocessing_function,
            shuffle=False,
            augment=False,
        )
        if self.ds_test is not None:
            self.ds_attack_test = self.prepare_ds(
                self.ds_test,
                cache=True,
                resize_rescale=True,
                img_shape=self.model_img_shape,
                batch_size=1,
                convert_to_rgb=self.convert_to_rgb,
                preprocessing_func=self.preprocessing_function,
                shuffle=False,
                augment=False,
            )

        self.ds_train = self.prepare_ds(
            self.ds_train,
            cache=True,
            resize_rescale=True,
            img_shape=self.model_img_shape,
            batch_size=self.batch_size,
            convert_to_rgb=self.convert_to_rgb,
            preprocessing_func=self.preprocessing_function,
            shuffle=self.shuffle,
            augment=self.augment_train,
        )

        if self.ds_val is not None:
            self.ds_val = self.prepare_ds(
                self.ds_val,
                cache=True,
                resize_rescale=True,
                img_shape=self.model_img_shape,
                batch_size=self.batch_size,
                convert_to_rgb=self.convert_to_rgb,
                preprocessing_func=self.preprocessing_function,
                shuffle=False,
                augment=False,
            )

        if self.ds_test is not None:
            self.ds_test = self.prepare_ds(
                self.ds_test,
                cache=True,
                resize_rescale=True,
                img_shape=self.model_img_shape,
                batch_size=self.batch_size,
                convert_to_rgb=self.convert_to_rgb,
                preprocessing_func=self.preprocessing_function,
                shuffle=False,
                augment=False,
            )

    def prepare_ds(
        self,
        ds: tf.data.Dataset,
        resize_rescale: bool,
        img_shape: Tuple[int, int, int],
        batch_size: Optional[int],
        convert_to_rgb: bool,
        preprocessing_func: Optional[Callable[[float], tf.Tensor]],
        shuffle: bool,
        augment: bool,
        cache: Union[str, bool] = True,
    ) -> tf.data.Dataset:
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
            preprocessing_layers.add(Rescaling(scale=1.0 / 255.0))

        if preprocessing_func:
            preprocessing_layers.add(ModelPreprocessing(preprocessing_func))

        if convert_to_rgb or resize_rescale or preprocessing_func:
            ds = ds.map(
                lambda x, y: (preprocessing_layers(x), y), num_parallel_calls=AUTOTUNE
            )

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(
                buffer_size=ds.cardinality().numpy(),
                seed=self.random_seed,
                reshuffle_each_iteration=False,
            )  # the each_iteration flag is really important to have a usable f1-score

        if batch_size is not None:
            ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)

        if augment:
            augmentation_layers = tf.keras.models.Sequential()

            if self.random_flip:
                augmentation_layers.add(RandomFlip(self.random_flip))

            if self.random_rotation:
                augmentation_layers.add(
                    RandomRotation(self.random_rotation, fill_mode="constant")
                )

            if self.random_translation_width and self.random_translation_height:
                augmentation_layers.add(
                    RandomTranslation(
                        self.random_translation_height,
                        self.random_translation_width,
                        fill_mode="constant",
                    )
                )
            if self.random_zoom:
                augmentation_layers.add(
                    RandomZoom(self.random_zoom, fill_mode="constant")
                )

            if self.random_brightness:
                augmentation_layers.add(RandomBrightness(self.random_brightness))

            ds = ds.map(
                lambda x, y: (augmentation_layers(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    def calculate_class_weights(
        self,
    ) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, float]]]:
        """Calculate class weights and class counts for train dataset."""
        class_labels, class_counts, class_distribution = self.get_class_distribution()

        class_counts_dict: Dict[str, int] = {}
        for y, count in zip(class_labels, class_counts):
            if self.class_names is not None and len(self.class_names) == len(
                class_labels
            ):
                class_counts_dict[f"{self.class_names[y]}({y})"] = count
            else:
                class_counts_dict[y] = count

        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(class_distribution),
            y=class_distribution,
        )

        class_weights: Dict[str, float] = {}
        if self.class_names is not None and len(self.class_names) == len(class_labels):
            for i, weight in enumerate(weights):
                class_weights[f"{self.class_names[y]}({y})"] = weight
        else:
            class_weights = dict(enumerate(weights))
        return class_counts_dict, class_weights

    def get_data_histogram(self, use_mean: bool = False) -> Tuple[np.array, np.array]:
        """Calculate histogram from train datasets.

        Return:
        ------
        Tuple[np.array, np.narray] -> (hist, bins)

        """
        samples = np.array(
            [sample for (sample, _) in list(tfds.as_numpy(self.ds_train))]
        )

        if use_mean:
            samples = np.mean(samples, axis=(0))

        hist, bins = np.histogram(samples, bins=range(255), density=True)
        return hist, bins

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

    def get_class_distribution(
        self, ds: Optional[tf.data.Dataset] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and return absolute class distribution from train dataset.

        This function returns the desired class_labels, class_counts and class_distribution values but also sets these variables as class variables.

        Parameter:
        --------
        ds: tf.data.Dataset - an optional dataset can be given to this function to calculate the class distribution of the given datasets
                              if ds is not set, it is assumed to calculate the class distribution from the current train dataset

        Return:
        ------
        (np.ndarray, np.ndarray, np.ndarray): three numpy arrays
            -> first one containing the class number
            -> second one containing the number of datapoints in the class (ordered)
            -> third one as a class representation for all datapoints
        f.e.: ([1,2,3,4,5],[404,133,313,122,10], [4,1,0,2,5,4,1,4,3,2,4,3,3,1,...])

        """
        if ds is not None:
            y_train = np.fromiter(ds.map(lambda _, y: y), int)
        else:
            y_train = np.fromiter(self.ds_train.map(lambda _, y: y), int)

        distribution = np.unique(y_train, return_counts=True)

        self.class_labels = distribution[0]
        self.class_counts = distribution[1]
        self.class_distribution = y_train

        return distribution + (y_train,)

    def calculate_class_imbalance2(self, ds: tf.data.Dataset) -> float:
        # Convert dataset to numpy arrays
        _, labels = zip(*list(ds.as_numpy_iterator()))
        labels = np.array(labels)

        # Calculate class distribution
        class_counts = np.bincount(labels)
        total_samples = len(labels)

        # Calculate class imbalance ratio
        class_imbalance_ratio = np.max(class_counts) / total_samples

        return class_imbalance_ratio

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

    def calculate_piqe_score(self) -> Dict[int, np.array]:
        """Calculate Perception-based Image QUality Evaluator (PIQUE) score without reference on train dataset."""
        class_dict: Dict[int, List[float]] = defaultdict(list)

        counter = 0
        for img, label in self.ds_train:
            label = int(label.numpy().astype("uint8"))
            img = img.numpy().astype("uint8")

            # cut off last dimension if grayscale image
            if img.shape[2] == 1:
                img = img[:, :, 0]

            score, _, _, _ = piqe(img)

            class_dict[label].append(score)

            counter += 1
            if counter % 100 == 0:
                print(f"Calculated PIQE score of {counter} images")

        return class_dict

    def calculate_compressed_image_size(self) -> Dict[int, np.array]:
        """Calculate compressed image size of all train dataset images.

        This function needs to be called before preprocessing the dataset.

        Returns a dict of numpy array mapped to classes.
        The numpy array contain for each image the following values in this order: entropy, uncompressed_size, png_size, png_ratio, jpeg_size, jpeg_ratio

        """
        class_dict: Dict[int, np.array] = {}

        counter = 0
        for img, label in self.ds_train:
            label = int(label.numpy().astype("uint8"))
            # check if grayscale or color image
            if img.shape[2] == 1:
                compressed_img = PIL.Image.fromarray(
                    img[:, :, 0].numpy().astype("uint8")
                )
            else:
                compressed_img = PIL.Image.fromarray(img.numpy().astype("uint8"))

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

            values = np.array(
                [entropy, uncompressed_size, png_size, png_ratio, jpeg_size, jpeg_ratio]
            )

            if label not in class_dict:
                class_dict[label] = values
            else:
                current = class_dict[label]
                class_dict[label] = np.vstack((current, values))

            counter += 1
            if counter % 100 == 0:
                print(f"Calculated Compression rates of {counter} images")

        return class_dict

    def calculate_image_fractal_dimension(self) -> Dict[int, List[float]]:
        """Calculate the fractal dimension of all images."""
        print("Calculating fractal dimension of all images")

        counter = 0
        class_dict: Dict[int, List[float]] = defaultdict(list)
        for img, label in self.ds_train:
            label = int(label.numpy().astype("uint8"))

            # check if grayscale or color image, convert to grayscale if RGB
            if img.shape[2] == 1:
                img = img.numpy().astype("uint8")[:, :, 0]
            else:
                img = tf.image.rgb_to_grayscale(img)
                img = img.numpy().astype("uint8")[:, :, 0]

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
                for row in image[
                    i : i + L
                ]:  # boxes that exceed bounds are shrunk to fit
                    for pixel in row[i : i + L]:
                        # lowest box is at G_min and each is h gray levels tall
                        height = (pixel - G_min) // h
                        boxes[height].append(
                            pixel
                        )  # assign the pixel intensity to the correct box
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

    def calculate_fdr(self) -> float:
        values = []
        labels = []
        for x, y in self.ds_train.as_numpy_iterator():
            values.append(
                x.reshape(-1)
            )  # transform (20, 20, 3) shape to (20*20*3) 1-D array shape
            labels.append(y)
        values = np.asarray(values)
        labels = np.asarray(labels)

        lda = LinearDiscriminantAnalysis()
        lda.fit(values, labels)

        between_class_eigenvalues = lda.explained_variance_ratio_
        within_class_eigenvalues = (
            lda.priors_ * (1 - lda.priors_) * (lda.means_**2).sum(axis=1)
        )

        fdr = np.sum(between_class_eigenvalues) / np.sum(within_class_eigenvalues)

        print(f"Fisher's Discriminant Ratio (FDR): {fdr}")
        return fdr

    def calculate_std(self) -> Dict[str, float]:
        """Calculate standard deviation of all dataset classes and for the whole dataset.

        Return:
        ------
        Dicŧ[str, float] - returns dict containing all classes std values and for the 'all' key the std value for the complete dataset

        """
        class_values: Dict[str, list] = defaultdict(list)
        all_values: List = []
        for x, y in self.ds_train.as_numpy_iterator():
            # combine all samples to one long list of values in a class
            # transform (20, 20, 3) shape to (20*20*3) 1-D array shape
            class_values[y].append(x.reshape(-1) / 255.0)
            all_values.append(x.reshape(-1) / 255.0)

        std_dict: Dict[str, float] = {}
        for k in class_values.keys():
            std_dict[int(k)] = np.asarray(class_values[k]).std()

        std_dict["all"] = np.asarray(all_values).std()
        return std_dict

    def build_ds_info(
        self,
        force_regeneration: bool = False,
        include_compression: bool = True,
        include_fract_dim: bool = True,
        include_piqe: bool = True,
        include_fdr: bool = True,
    ):
        """Build dataset info dictionary.

        This function needs to be called after initializing and loading the dataset,
        but before calling preprocessing on it!

        """

        # before building new ds_info try to load an existing one
        if not force_regeneration:
            self.load_ds_info_from_json()
            print(self.ds_info)

        self.ds_info["name"] = (str(self.dataset_name),)  # not useful for dataframe
        self.ds_info["dataset_img_shape"] = self.dataset_img_shape
        self.ds_info["model_img_shape"] = self.model_img_shape

        count_keys = set(
            [
                "total_count",
                "train_count",
                "val_count",
                "test_count",
                "num_classes",
                "class_weights",
                "class_counts",
            ]
        )
        if not count_keys.issubset(set(self.ds_info.keys())):
            class_counts, class_weights = self.calculate_class_weights()
            ds_count = self.get_dataset_count()
            total_count: int = sum(ds_count.values())
            # convert int64 keys to int keys -> to jsonify
            class_counts = {str(k): int(v) for k, v in class_counts.items()}
            class_weights = {str(k): float(v) for k, v in class_weights.items()}
            self.ds_info["total_count"] = total_count
            self.ds_info["train_count"] = ds_count["train"]
            self.ds_info["val_count"] = ds_count["val"]
            self.ds_info["test_count"] = ds_count["test"]
            self.ds_info["num_classes"] = self.num_classes
            self.ds_info["class_weights"] = class_weights  # not useful for dataframe
            self.ds_info["class_counts"] = class_counts  # not useful for dataframe

        pil_keys = set(
            [
                "avg_png_size",
                "avg_jpeg_size",
                "avg_png_ratio",
                "avg_jpeg_ratio",
                "avg_entropy",
                "avg_byte_count",
                "clas_avg_entropy",
                "class_avg_png_size",
                "class_avg_jpeg_size",
                "class_avg_png_ratio",
                "class_avg_jpeg_ratio",
            ]
        )
        if (not pil_keys.issubset(set(self.ds_info.keys()))) and include_compression:
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

                avg_class_entropy[int(k)] = avg_entropy
                avg_class_png_size[int(k)] = avg_png_size
                avg_class_png_ratio[int(k)] = avg_png_ratio
                avg_class_jpeg_size[int(k)] = avg_jpeg_size
                avg_class_jpeg_ratio[int(k)] = avg_jpeg_ratio

            self.ds_info["avg_byte_count"] = (avg_ds_byte_size,)
            self.ds_info["avg_entropy"] = (avg_ds_entropy,)
            self.ds_info["avg_png_size"] = (avg_ds_png_size,)
            self.ds_info["avg_png_ratio"] = (avg_ds_png_ratio,)
            self.ds_info["avg_jpeg_size"] = (avg_ds_jpeg_size,)
            self.ds_info["avg_jpeg_ratio"] = (avg_ds_jpeg_ratio,)
            self.ds_info["class_avg_entropy"] = (
                avg_class_entropy,
            )  # not useful for dataframe
            self.ds_info["class_avg_png_size"] = (
                avg_class_png_size,
            )  # not useful for dataframe
            self.ds_info["class_avg_png_ratio"] = (
                avg_class_png_ratio,
            )  # not useful for dataframe
            self.ds_info["class_avg_jpeg_size"] = (
                avg_class_jpeg_size,
            )  # not useful for dataframe
            self.ds_info["class_avg_jpeg_ratio"] = (
                avg_class_jpeg_ratio,
            )  # not useful for dataframe

        if "class_imbalance" not in self.ds_info:
            # not useful for dataframe
            self.ds_info["class_imbalance"] = self.calculate_class_imbalance()

        if (
            ("class_avg_fractal_dim" not in self.ds_info)
            or ("avg_fractal_dim" not in self.ds_info)
        ) and include_fract_dim:
            avg_class_fractal_dim: Dict[int, float] = {}
            avg_ds_fractal_dim: float = 0.0
            fractal_dim_dict = self.calculate_image_fractal_dimension()

            for k, v in fractal_dim_dict.items():
                avg_ds_fractal_dim += np.sum(v)
                avg_class_fractal_dim[k] = np.mean(v)

            avg_ds_fractal_dim = avg_ds_fractal_dim / len(
                [item for sublist in fractal_dim_dict.values() for item in sublist]
            )

            self.ds_info["avg_fractal_dim"] = avg_ds_fractal_dim
            self.ds_info["class_avg_fractal_dim"] = avg_class_fractal_dim

        if (
            "class_avg_piqe" not in self.ds_info or "avg_piqe" not in self.ds_info
        ) and include_piqe:
            avg_class_piqe: Dict[int, float] = {}
            avg_ds_piqe: float = 0.0
            piqe_dict = self.calculate_piqe_score()

            for k, v in piqe_dict.items():
                avg_ds_piqe += np.sum(v)
                avg_class_piqe[k] = np.mean(v)

            avg_ds_piqe = avg_ds_piqe / len(
                [item for sublist in piqe_dict.values() for item in sublist]
            )

            self.ds_info["avg_piqe"] = avg_ds_piqe
            self.ds_info["class_avg_piqe"] = avg_class_piqe

        if "fdr" not in self.ds_info and include_fdr:
            fdr = self.calculate_fdr()
            self.ds_info["fdr"] = fdr

        if "class_std" not in self.ds_info or "std" not in self.ds_info:
            std_dict = self.calculate_std()
            self.ds_info["std"] = std_dict["all"]
            # the "all" entry contains the std for the complete dataset, if we remove this entry, only the per-class std is left
            del std_dict["all"]
            self.ds_info["class_std"] = std_dict

        # prettify ds info dict
        for k, v in self.ds_info.items():
            if isinstance(v, tuple):
                if len(v) == 1:
                    # unpack tupled values
                    (self.ds_info[k],) = v
        print(self.ds_info)

    def get_ds_info_as_df(self) -> pd.DataFrame:
        # modify dict to make suitable dataframe from it
        dict_cpy = self.ds_info.copy()
        del dict_cpy["class_counts"]
        del dict_cpy["class_weights"]
        del dict_cpy["class_avg_entropy"]
        del dict_cpy["class_avg_png_size"]
        del dict_cpy["class_avg_png_ratio"]
        del dict_cpy["class_avg_jpeg_size"]
        del dict_cpy["class_avg_jpeg_ratio"]
        del dict_cpy["class_avg_fractal_dim"]
        del dict_cpy["class_avg_piqe"]
        del dict_cpy["class_std"]
        del dict_cpy["name"]

        dict_cpy["dataset_img_shape"] = "/".join(
            map(str, dict_cpy["dataset_img_shape"])
        )
        dict_cpy["model_img_shape"] = "/".join(map(str, dict_cpy["model_img_shape"]))

        df = pd.DataFrame(dict_cpy, index=[self.dataset_name])

        return df

    def save_ds_info_as_json(self):
        """Save the ds_info dictionary to a json file."""
        ds_info_json_file = os.path.join(
            self.ds_info_path, f"{self.dataset_name}_ds_info.json"
        )
        save_dict_as_json(self.ds_info, ds_info_json_file)

    def load_ds_info_from_json(self):
        """Load the ds_info dictionary from a json file."""
        ds_info_json_file = os.path.join(
            self.ds_info_path, f"{self.dataset_name}_ds_info.json"
        )
        if not os.path.isfile(ds_info_json_file):
            print(
                f"Cannot load ds-info dict from json, since json file {ds_info_json_file} does not exists"
            )
            return
        self.ds_info = load_dict_from_json(ds_info_json_file)

    def get_train_ds_subset(
        self, keep: np.ndarray, apply_processing: bool = False
    ) -> tf.data.Dataset:
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
            ds = self.prepare_ds(
                ds,
                cache=True,
                resize_rescale=True,
                img_shape=self.model_img_shape,
                batch_size=self.batch_size,
                convert_to_rgb=self.convert_to_rgb,
                preprocessing_func=self.preprocessing_function,
                shuffle=self.shuffle,
                augment=self.augment_train,
            )
        else:
            ds = (
                ds.cache()
                .batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        return ds

    def get_train_ds_as_numpy(
        self, unbatch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_train, unbatch=unbatch)

    def get_test_ds_as_numpy(
        self, unbatch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_test, unbatch)

    def get_val_ds_as_numpy(
        self, unbatch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Validation Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_val, unbatch)

    def get_attack_train_ds_as_numpy(
        self, unbatch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_train, unbatch)

    def get_attack_test_ds_as_numpy(
        self, unbatch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_test, unbatch)


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


@dataclass
class AbstractDatasetClassSize(AbstractDataset):
    class_size: int = field(init=False, repr=True)

    def reduce_samples_per_class_train_ds(self, max_samples_per_class: int) -> None:
        """Reduce all samples in the train_ds per class to a specific value.

        The train_ds is directly modified, no dataset copy is returned.

        Parameter:
        --------
        max_samples_per_class : int - the number of samples that all classes should get reduced to

        """
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
                tf.TensorSpec(shape=label.shape, dtype=label.dtype),
            ),
        )
        self.ds_train = reduced_dataset
        ds_len = sum(1 for _ in self.ds_train)
        self.ds_train = self.ds_train.apply(
            tf.data.experimental.assert_cardinality(ds_len)
        )
        print(f"Reduced class size to {max_samples_per_class}")

    def _load_dataset(self):
        print(
            f"Creating {self.dataset_name} dataset with class size of {self.class_size}"
        )
        self.reduce_samples_per_class_train_ds(self.class_size)


@dataclass
class AbstractDatasetClassImbalance(AbstractDataset):
    # can eiter be N - normal, L - linear
    imbalance_mode: str = field(init=False, repr=True)
    imbalance_ratio: str = field(init=False, repr=True)

    def make_unbalanced_dataset(
        self, ds: tf.data.Dataset, imbalance_ratio: float, distribution: str = "N"
    ):
        """Create an unbalanced dataset from a balanced one.

        Parameter:
        --------
        ds : dataset to be unbalanced
        imbalance_ratio : float[0-1] -  the unbalance factor to be applied,
                                        a value between 0 (lesser imbalance) and 1 (more imbalance)
        distribution : str -    either 'L' (linear) or 'N' (normal distribution), specifies how the dataset is resampled to introduce imbalance
                                lin -> the class count is reduced linearely starting from the class with the most samples in class: [100, 90, 80, 70, 60, 50], the imbalance factor specifies how much is subtracted in each iteration
                                norm -> te class count is reduced with a normal distribution: [50, 60, 70, 60, 50], the imbalance_ratio factor specifies the norm scale, the loc is set to 1-imbalance_ratio
        """
        # Convert balanced dataset to numpy arrays
        (values, labels) = get_ds_as_numpy(ds, unbatch=False)

        classes, class_count, _ = self.get_class_distribution(ds)

        # generate distribution based class imbalance
        if distribution == "L":
            classes_class_count = zip(classes, class_count)
            # sort classes by class count, first entry is largest class
            classes_class_count = sorted(
                classes_class_count, key=lambda x: x[1], reverse=True
            )

            # create the imbalanced class counts
            # the smallest class size is multiplied with the imbalance_ratio factor, therefore decreasing its size
            new_class_counts = np.linspace(
                classes_class_count[0][1],
                int(classes_class_count[-1][1] * (1 - imbalance_ratio)),
                num=len(classes_class_count),
            )

            class_count_dict = {}
            for i, (class_count) in enumerate(classes_class_count):
                # only reduce class size if is not already smaller than the newly calculated value for it
                if class_count[1] > new_class_counts[i]:
                    class_count_dict[class_count[0]] = int(new_class_counts[i])
                else:
                    class_count_dict[class_count[0]] = class_count[1]

            d1, d2, d3, d4 = values.shape
            values = values.reshape((d1, d2 * d3 * d4))
            (values, labels) = make_imbalance(
                X=values,
                y=labels,
                sampling_strategy=class_count_dict,
                random_state=self.random_seed,
            )
            d1, _ = values.shape
            values = values.reshape((d1, d2, d3, d4))
            return tf.data.Dataset.from_tensor_slices((values, labels))

        elif distribution == "N":
            class_count_dict_path = os.path.join(
                self.ds_info_path, f"{self.dataset_name}_class_count_dict.json"
            )
            # check if we can load a previous class_count_dict, and then load it
            class_count_dict = load_dict_from_json(class_count_dict_path)
            if class_count_dict is None:
                # create new class_count_dict if we could not load it
                random_array = np.random.normal(
                    loc=1 - imbalance_ratio,
                    scale=imbalance_ratio,
                    size=len(class_count),
                )

                # clip array to prevent values greater 1 or too small values
                random_array = np.clip(random_array, 0.05, 1.0)
                class_count = (class_count * random_array).astype(int)
                class_count_dict = dict(zip(classes, class_count))

            # convert dict keys to int
            class_count_dict = {int(k): v for k, v in class_count_dict.items()}

            d1, d2, d3, d4 = values.shape
            values = values.reshape((d1, d2 * d3 * d4))
            (values, labels) = make_imbalance(
                X=values,
                y=labels,
                sampling_strategy=class_count_dict,
                random_state=self.random_seed,
            )

            d1, _ = values.shape
            values = values.reshape((d1, d2, d3, d4))

            # save the class_count dict to reproduce this dataset
            save_dict_as_json(class_count_dict, class_count_dict_path)
            return tf.data.Dataset.from_tensor_slices((values, labels))

        else:
            print(
                f"distribution {distribution} is not implemented! Cannot imbalance dataset."
            )
            return None

    def _load_dataset(self):
        print(
            f"Creating {self.dataset_name} dataset with imbalance of {self.imbalance_ratio} in {self.imbalance_mode} mode"
        )
        self.ds_train = self.make_unbalanced_dataset(
            self.ds_train, self.imbalance_ratio, distribution=self.imbalance_mode
        )


@dataclass
class AbstractDatasetGray(AbstractDataset):
    def convertds_to_grayscale(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        to_grayscale = tf.keras.Sequential([RgbToGrayscale()])
        ds = ds.map(lambda x, y: (to_grayscale(x, training=True), y))
        return ds

    def _load_dataset(self):
        print(f"Creating {self.dataset_name} dataset with only grayscale images")

        if self.ds_test is not None:
            self.ds_test = self.convertds_to_grayscale(self.ds_test)
        if self.ds_val is not None:
            self.ds_val = self.convertds_to_grayscale(self.ds_val)
        self.ds_train = self.convertds_to_grayscale(self.ds_train)


@dataclass
class AbstractDatasetCustomClasses(AbstractDataset):
    kept_classes: List[int] = field(init=False, repr=True)
    new_num_classes: int = field(init=False, repr=True)

    def reduce_classes(self, classes_to_keep: List[int], ds: tf.data.Dataset):
        new_samples = []

        for sample, label in ds:
            if label in classes_to_keep:
                new_samples.append((sample, label))

        new_ds = tf.data.Dataset.from_generator(
            lambda: (sample_label for sample_label in new_samples),
            output_signature=(
                tf.TensorSpec(shape=sample.shape, dtype=sample.dtype),
                tf.TensorSpec(shape=label.shape, dtype=label.dtype),
            ),
        )
        ds_len = sum(1 for _ in new_ds)
        new_ds = new_ds.apply(tf.data.experimental.assert_cardinality(ds_len))
        return new_ds

    def _load_dataset(self):
        # chose which classes and labels to keep
        classes, _, _ = self.get_class_distribution()

        if self.new_num_classes > len(classes):
            print(
                "ERROR: Cannot set a higher number of classes to set than already existant in dataset! Current class count: {len(classes)} <-> desired class count: {self.new_num_classes}"
            )
            sys.exit(1)

        self.kept_classes = classes[: self.new_num_classes]

        print(
            f"Creating {self.dataset_name} dataset with changed number of classes ({self.new_num_classes} classes, which are classes with labels: {self.kept_classes}.)"
        )

        if self.ds_test is not None:
            self.ds_test = self.reduce_classes(self.kept_classes, self.ds_test)
        if self.ds_val is not None:
            self.ds_val = self.reduce_classes(self.kept_classes, self.ds_val)
        self.ds_train = self.reduce_classes(self.kept_classes, self.ds_train)
