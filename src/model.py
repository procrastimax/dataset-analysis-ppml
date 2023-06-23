from ppml_datasets.utils import check_create_folder, visualize_training
import tensorflow as tf
from tensorflow_privacy import VectorizedDPKerasAdamOptimizer
from tensorflow import keras
from keras.callbacks import EarlyStopping
from util import compute_delta, compute_noise, compute_dp_sgd_privacy
import numpy as np
import os
from abc import ABC, abstractmethod

from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Model(ABC):
    img_height: int
    img_width: int
    color_channels: int
    num_classes: int
    batch_size: int
    epochs: int

    model_path: str
    model_name: str

    momentum: float
    learning_rate: float
    epochs: int
    use_early_stopping: bool
    patience: int

    is_private_model: bool

    l2_norm_clip: float = field(init=False, default=None)
    noise_multiplier: float = field(init=False, default=None)
    num_microbatches: float = field(init=False, default=None)
    epsilon: float = field(init=False, default=None)

    model: Optional[keras.Sequential] = field(init=False, default=None)
    history: Optional[tf.keras.callbacks.History] = field(init=False, default=None)

    @abstractmethod
    def build_model(self):
        # Has to be implemented by child class
        # Remember to implement horizontal flipping augmentation
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    def set_privacy_parameter(self, epsilon: float, num_train_samples: int, l2_norm_clip: float, num_microbatches: int):
        delta = compute_delta(num_train_samples)
        self.noise_multiplier = compute_noise(
            num_train_samples, self.batch_size, epsilon, self.epochs, delta)
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches

        # calculate epsilon to verify calculated noise values
        calc_epsilon = compute_dp_sgd_privacy(num_train_samples, self.batch_size,
                                              self.noise_multiplier, self.epochs, delta)
        print(f"Calculated epsilon is {calc_epsilon}")

    def build_compile(self):
        """Build and compile CNN model."""
        self.build_model()
        self.compile_model()

    def print_summary(self):
        print("Model summary:")
        self.model.summary()

    def train_model_from_ds(self,
                            train_ds: tf.data.Dataset,
                            val_ds: tf.data.Dataset) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=self.patience, restore_best_weights=False)
            callback_list.append(es)

        ds_len = len(list(train_ds.unbatch().as_numpy_iterator()))
        steps_per_epoch = ds_len // self.batch_size

        self.history = self.model.fit(x=train_ds,
                                      validation_data=val_ds,
                                      epochs=self.epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=callback_list)

        return self.history

    def train_model_from_numpy(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               batch: int,
                               val_x: Optional[np.ndarray] = None,
                               val_y: Optional[np.ndarray] = None,
                               val_split: Optional[float] = None) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                               patience=self.patience, restore_best_weights=True)
            callback_list.append(es)

        validation_data = None
        if val_x is not None and val_y is not None:
            validation_data = (val_x, val_y)

        steps_per_epoch = len(x) // self.batch_size

        self.history = self.model.fit(x=x, y=y, validation_data=validation_data,
                                      validation_split=val_split,
                                      epochs=self.epochs,
                                      batch_size=batch,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=callback_list)
        return self.history

    def save_train_history(self, folder_name: str, image_name: str):
        """Save training history as image.

        Creates folder if folder_name folder does not exists.
        """
        print("Saving shadow model train history as figure")
        check_create_folder(folder_name)
        visualize_training(history=self.history, img_name=os.path.join(
            folder_name, image_name))

    def test_model(self, test_ds: tf.data.Dataset) -> Tuple[float, float]:
        """Run the model's prediction function on the given tf.data.Dataset.

        Return:
        ------
        Tuple[float, float] -> (loss, accuracy)

        """
        test_loss, test_acc = self.model.evaluate(x=test_ds)
        print(f"\nAccuracy: {test_acc}, Loss: {test_loss}")
        return (test_loss, test_acc)

    def save_model(self):
        print(f"Saving {self.model_name} model to {self.model_path}")
        self.model.save(filepath=self.model_path, save_format="h5", overwrite=True)

    def load_model(self):
        """Load from model filepath.

        The model is loaded uncompiled, so the model has to be compiled after loading it.
        """
        self.model = tf.keras.models.load_model(filepath=self.model_path, compile=False)


@ dataclass
class SmallCNNModel(Model):
    def build_model(self):
        print("Building model")
        model = tf.keras.models.Sequential()
        # Add a layer to do random horizontal augmentation.
        model.add(tf.keras.layers.RandomFlip('horizontal',
                                             input_shape=(self.img_height, self.img_width, 3)))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10))

        self.model = None  # reset previous model
        self.model = model

    def compile_model(self):
        print("Compiling model")
        optimizer = self.get_optimizer()
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"])

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
        )


@ dataclass
class PrivateSmallCNNModel(Model):
    def build_model(self):
        print("Building model")
        model = tf.keras.models.Sequential()
        # Add a layer to do random horizontal augmentation.
        model.add(tf.keras.layers.RandomFlip('horizontal',
                                             input_shape=(self.img_height, self.img_width, 3)))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10))

        self.model = None  # reset previous model
        self.model = model

    def compile_model(self):
        print("Compiling model")
        optimizer: tf.keras.optimizers.Optimizer = self.get_optimizer()

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.losses.Reduction.NONE)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"])

    def get_optimizer(self):
        optimizer = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.num_microbatches,
            learning_rate=self.learning_rate,
        )
        return optimizer
