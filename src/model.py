from ppml_datasets.utils import check_create_folder, visualize_training
import tensorflow as tf
from tensorflow_privacy import VectorizedDPKerasAdamOptimizer
from tensorflow import keras
from keras.callbacks import EarlyStopping
from util import compute_delta, compute_noise, compute_privacy
import numpy as np
import os
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report

from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class Model(ABC):
    img_height: int
    img_width: int
    color_channels: int
    num_classes: int
    batch_size: int
    epochs: int
    random_seed: int

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
        used_microbatching = True
        if num_microbatches == 1:
            used_microbatching = False
        self.noise_multiplier = compute_noise(
            num_train_samples=num_train_samples,
            batch_size=self.batch_size,
            target_epsilon=epsilon,
            epochs=self.epochs,
            delta=delta)
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches

        # calculate epsilon to verify calculated noise values
        calc_epsilon = compute_privacy(n=num_train_samples,
                                       batch_size=self.batch_size,
                                       noise_multiplier=self.noise_multiplier,
                                       epochs=self.epochs,
                                       delta=delta,
                                       used_microbatching=used_microbatching)
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
                               patience=self.patience, restore_best_weights=True)
            callback_list.append(es)
        # create new DS and cut off the samples that did not fit into a batch
        train_ds = train_ds.unbatch()
        ds_len = len(list(train_ds.as_numpy_iterator()))
        steps_per_epoch = ds_len // self.batch_size
        train_ds = train_ds.repeat()
        train_ds = train_ds.take(self.batch_size * steps_per_epoch)
        train_ds = train_ds.batch(self.batch_size)

        self.history = self.model.fit(x=train_ds,
                                      validation_data=val_ds,
                                      epochs=self.epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=callback_list)

        return self.history

    def train_model_from_numpy(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               val_x: np.ndarray,
                               val_y: np.ndarray) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                               patience=self.patience, restore_best_weights=True)
            callback_list.append(es)

        val_y = tf.one_hot(indices=val_y, depth=self.num_classes)
        validation_data = (val_x, val_y)

        steps_per_epoch = len(x) // self.batch_size

        # create new numpy array and cut off the samples that did not fit into a batch
        x = x[:self.batch_size * steps_per_epoch]
        y = y[:self.batch_size * steps_per_epoch]
        y = tf.one_hot(indices=y, depth=self.num_classes)

        self.history = self.model.fit(x=x,
                                      y=y,
                                      validation_data=validation_data,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
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

    def test_model(self, ds: tf.data.Dataset) -> Dict[str, float]:
        """Run the model's prediction function on the given tf.data.Dataset.

        Return:
        ------
        Tuple[float, float] -> (loss, accuracy)

        """

        loss, acc = self.model.evaluate(x=ds, verbose=2)

        labels = []
        for sample, label in ds.unbatch().as_numpy_iterator():
            labels.append(label)

        performance_results = {"loss": loss,
                               "accuracy": acc}

        pred = self.model.predict(ds, verbose=2)
        pred = tf.argmax(pred, axis=1)

        report_dict = classification_report(y_true=labels, y_pred=pred, output_dict=True)
        print(classification_report(y_true=labels, y_pred=pred, output_dict=False, digits=3))

        performance_results.update({
            'precision': report_dict['macro avg']['precision'],
            'recall': report_dict['macro avg']['recall'],
            'f1-score': report_dict['macro avg']['f1-score'],
        })

        return performance_results

    def save_model(self):
        print(f"Saving {self.model_name} model to {self.model_path}")
        self.model.save(filepath=self.model_path, save_format="keras", overwrite=True)

    def load_model(self):
        """Load from model filepath.

        The model is loaded uncompiled, so the model has to be compiled after loading it.
        """
        self.model = tf.keras.models.load_model(filepath=self.model_path, compile=False)

    def build_model(self):
        print("Building model")
        model = tf.keras.models.Sequential()
        # Add a layer to do random horizontal augmentation.
        model.add(tf.keras.layers.RandomFlip('horizontal',
                                             seed=self.random_seed,
                                             input_shape=(self.img_height, self.img_width, 3)))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
            3, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(self.num_classes))

        self.model = None  # reset previous model
        self.model = model


@ dataclass
class CNNModel(Model):
    def compile_model(self):
        print("Compiling model")
        optimizer = self.get_optimizer()
        print(f"num classes: {self.num_classes}")
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
class PrivateCNNModel(Model):
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
