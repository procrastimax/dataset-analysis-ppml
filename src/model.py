import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow_privacy import DPSequential
from tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry import (
    LayerRegistry,
    dense_layer_computation,
)

from ppml_datasets.utils import check_create_folder, visualize_training
from util import (
    compute_delta,
    compute_noise,
    compute_numerical_epsilon,
    compute_privacy,
)


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

    learning_rate: float
    epochs: int

    # default value according to: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    adam_epsilon: float = 1e-7

    ema_momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    use_early_stopping: Optional[bool] = None
    patience: Optional[int] = None

    l2_norm_clip: float = field(init=False, default=None)
    noise_multiplier: float = field(init=False, default=None)
    num_microbatches: float = field(init=False, default=None)
    epsilon: float = field(init=False, default=None)
    delta: float = field(init=False, default=None)

    model: Optional[tf.keras.Sequential] = field(init=False, default=None)
    history: Optional[tf.keras.callbacks.History] = field(init=False, default=None)

    @abstractmethod
    def compile_model(self):
        # Has to be implemented by child class
        # Remember to implement horizontal flipping augmentation
        pass

    @abstractmethod
    def get_optimizer(self):
        # Has to be implemented by child class
        # Remember to implement horizontal flipping augmentation
        pass

    @abstractmethod
    def build_model(self):
        # Has to be implemented by child class
        # Remember to implement horizontal flipping augmentation
        pass

    def set_privacy_parameter(
        self,
        epsilon: float,
        num_train_samples: int,
        l2_norm_clip: float,
        num_microbatches: int,
        noise_multiplier: Optional[float] = None,
    ):
        self.delta = compute_delta(num_train_samples)
        self.epsilon = epsilon
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches

        used_microbatching = True
        if num_microbatches == 1:
            used_microbatching = False

        # calc noise_multiplier if not already set
        if noise_multiplier is None:
            self.noise_multiplier = compute_noise(
                num_train_samples=num_train_samples,
                batch_size=self.batch_size,
                target_epsilon=epsilon,
                epochs=self.epochs,
                delta=self.delta,
            )
        else:
            self.noise_multiplier = noise_multiplier

        # calculate epsilon to verify calculated noise values
        calc_epsilon = compute_privacy(
            n=num_train_samples,
            batch_size=self.batch_size,
            noise_multiplier=self.noise_multiplier,
            epochs=self.epochs,
            delta=self.delta,
            used_microbatching=used_microbatching,
        )
        print(f"Calculated epsilon is {calc_epsilon}")

        steps = self.epochs * num_train_samples // self.batch_size
        numerical_epsilon = compute_numerical_epsilon(
            steps=steps,
            noise_multiplier=self.noise_multiplier,
            batch_size=self.batch_size,
            num_samples=num_train_samples,
        )
        print(f"Another calculated epsilon is: {numerical_epsilon}")

    def build_compile(self):
        """Build and compile CNN model."""
        self.build_model()
        self.compile_model()

    def print_summary(self):
        print("Model summary:")
        self.model.summary()

    def train_model_from_ds(
        self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset
    ) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=self.patience,
                restore_best_weights=True,
            )
            callback_list.append(es)
        # create new DS and cut off the samples that did not fit into a batch
        train_ds = train_ds.unbatch()
        ds_len = len(list(train_ds.as_numpy_iterator()))
        steps_per_epoch = ds_len // self.batch_size
        train_ds = train_ds.repeat()
        train_ds = train_ds.take(self.batch_size * steps_per_epoch)
        train_ds = train_ds.batch(self.batch_size)

        self.history = self.model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=2,
        )

        return self.history

    def train_model_from_numpy(
        self, x: np.ndarray, y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
    ) -> tf.keras.callbacks.History:
        callback_list = []
        if self.use_early_stopping:
            es = EarlyStopping(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=self.patience,
                restore_best_weights=True,
            )
            callback_list.append(es)

        validation_data = (val_x, val_y)
        steps_per_epoch = len(x) // self.batch_size

        # create new numpy array and cut off the samples that did not fit into a batch
        x = x[: self.batch_size * steps_per_epoch]
        y = y[: self.batch_size * steps_per_epoch]

        self.history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=2,
        )
        return self.history

    def save_train_history(self, folder_name: str, image_name: str):
        """Save training history as image.

        Creates folder if folder_name folder does not exists.
        """
        print("Saving shadow model train history as figure")
        check_create_folder(folder_name)
        visualize_training(
            history=self.history, img_name=os.path.join(folder_name, image_name)
        )

    def test_model(self, x: np.array, y: np.array) -> Dict[str, Any]:
        """Run the model's prediction function on the given tf.data.Dataset.

        Return:
        ------
        Dict[str, Any]

        """
        loss, acc = self.model.evaluate(x, y, batch_size=self.batch_size, verbose=2)

        performance_results = {"loss": loss, "accuracy": acc}
        # F1-Scores of every class
        class_performance_results: Dict[str, float] = {}

        pred = self.model.predict(x, batch_size=self.batch_size, verbose=2)
        # find likeliest class
        pred = tf.argmax(pred, axis=1)
        # convert to one-hot encoding
        pred = tf.one_hot(pred, depth=self.num_classes)

        report_dict = classification_report(y_true=y, y_pred=pred, output_dict=True)
        print(classification_report(y_true=y, y_pred=pred, output_dict=False, digits=3))

        performance_results.update(
            {
                "precision": report_dict["macro avg"]["precision"],
                "recall": report_dict["macro avg"]["recall"],
                "f1-score": report_dict["macro avg"]["f1-score"],
            }
        )

        for k, v in report_dict.items():
            # all class performances start with the class number
            if k.isdigit():
                class_performance_results[k] = float(v["f1-score"])

        performance_results["class-wise"] = class_performance_results

        return performance_results

    def save_model(self):
        print(f"Saving {self.model_name} model to {self.model_path}")
        self.model.save_weights(
            filepath=self.model_path, overwrite=True, save_format="tf"
        )

    def load_model(self):
        """Load from model filepath.

        The model is loaded uncompiled, so the model has to be compiled after loading it.
        """
        print(f"Loading model from {self.model_path}")
        self.build_compile()
        self.model.load_weights(filepath=self.model_path)

    def get_layer(self) -> List[tf.keras.layers.Layer]:
        groups = 32
        return [
            tf.keras.layers.RandomFlip(
                "horizontal",
                seed=self.random_seed,
                input_shape=(self.img_height, self.img_width, 3),
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.GroupNormalization(groups=groups),
            tf.keras.layers.Dense(self.num_classes),
        ]


@dataclass
class CNNModel(Model):
    def compile_model(self):
        print("Compiling model")
        optimizer = self.get_optimizer()
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
            use_ema=self.ema_momentum is not None,
            ema_momentum=self.ema_momentum,
            jit_compile=True,
        )

    def build_model(self):
        print("Building non-private model")
        self.model = tf.keras.Sequential(super().get_layer())


@dataclass
class PrivateCNNModel(Model):
    def compile_model(self):
        print("Compiling model")
        optimizer = self.get_optimizer()
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
            use_ema=self.ema_momentum is not None,
            ema_momentum=self.ema_momentum,
            jit_compile=True,
        )

    def build_model(self):
        print("Building private model")

        layer_registry = LayerRegistry()
        layer_registry.insert(tf.keras.layers.Dense, dense_layer_computation)

        self.model = DPSequential(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.num_microbatches,
            layers=super().get_layer(),
            layer_registry=layer_registry,
        )
