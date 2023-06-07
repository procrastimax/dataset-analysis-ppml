from ppml_datasets.utils import check_create_folder, visualize_training
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import numpy as np
import os
from abc import ABC, abstractmethod

from typing import Optional, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class Model(ABC):
    img_height: int
    img_width: int
    color_channels: int
    num_classes: int
    batch_size: int
    epochs: int
    model_name: str
    model_path: str = "models"

    use_early_stopping: bool = True
    patience: int = 15

    model: Optional[keras.Sequential] = field(init=False, default=None)
    history: Optional[tf.keras.callbacks.History] = field(init=False, default=None)

    def __post_init__(self):
        self.model_path = os.path.join(self.model_path, self.model_name)
        check_create_folder(self.model_path)

    @abstractmethod
    def build_model(self):
        # Has to be implemented by child class
        # Remember to implement horizontal flipping augmentation
        pass

    @abstractmethod
    def compile_model(self):
        pass

    @abstractmethod
    def reset_model_optimizer(self):
        pass

    def build_compile(self):
        """Build and compile CNN model."""
        self.build_model()
        self.compile_model()

    def print_summary(self):
        print("Model summary:")
        self.model.summary()

    def train_model_from_ds(self,
                            train_ds: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
                            val_ds: Optional[Union[tf.data.Dataset]] = None) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=self.patience, restore_best_weights=False)
            callback_list.append(es)

        self.history = self.model.fit(x=train_ds, validation_data=val_ds,
                                      epochs=self.epochs, callbacks=callback_list)

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

        self.history = self.model.fit(x=x, y=y, validation_data=validation_data,
                                      validation_split=val_split,
                                      epochs=self.epochs,
                                      batch_size=batch,
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
        self.model = tf.keras.models.load_model(filepath=self.model_path)


@dataclass
class SmallCNNModel(Model):
    img_height: int
    img_width: int
    color_channels: int
    num_classes: int
    batch_size: int
    model_name: str = "small_cnn"
    model_path: str = "models"

    momentum: float = 0.9
    learning_rate: float = 0.02
    epochs: int = 500

    use_early_stopping: bool = True
    # used for EarlyStopping
    patience: int = 15

    model: Optional[keras.Sequential] = field(init=False, default=None)
    history: Optional[tf.keras.callbacks.History] = field(init=False, default=None)

    def build_model(self):
        print("Building model")
        model = tf.keras.models.Sequential()
        # Add a layer to do random horizontal augmentation.
        model.add(tf.keras.layers.RandomFlip('horizontal',
                  input_shape=(self.img_height, self.img_width, 3)))

        for _ in range(3):
            model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10))

        self.model = None
        self.model = model

    def compile_model(self):
        print("Compiling model")
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=self.momentum),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True), metrics=["accuracy"])

    def reset_model_optimizer(self):
        """Reset tensorflow model, optimizer and history by overwriting old instances with new initialization.

        Should only be called after a copy of CNNModel class was created.
        """
        self.model = keras.Sequential()
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )
        self.history = None
