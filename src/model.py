import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List, Optional

from dataclasses import dataclass, field


@dataclass
class CNNModel():
    img_height: int
    img_width: int
    color_channels: int
    num_classes: int
    batch_size: int
    model_path: str = "data/models/cnn_model"

    # sequential list of convolutional filter dimensions
    filter_dim_list: List[int] = field(default_factory=lambda: [16, 32, 64])
    # sequential list of convolutional kernel dimensions
    kernel_dim_list: List[int] = field(default_factory=lambda: [3, 3, 3])
    dense_layer_dimension: int = 128

    padding: str = "same"
    conv_activation: str = "relu"
    dense_activation: str = "relu"
    dropout: Optional[float] = None
    learning_rate: float = 0.01
    momentum: float = 0.0
    epochs: int = 50

    model: keras.Sequential = field(init=False, repr=False, default=keras.Sequential())
    history: Optional[tf.keras.callbacks.History] = field(init=False, default=None)
    optimizer: tf.keras.optimizers.Optimizer = field(init=False, default=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum))

    def build_model(self):
        print("Building model")

        self.model.add(keras.Input(shape=(self.img_height, self.img_width,
                                          self.color_channels), batch_size=self.batch_size))

        for (i, (filter_dim, kernel_dim)) in enumerate(zip(self.filter_dim_list,
                                                           self.kernel_dim_list)):
            self.model.add(layers.Conv2D(filter_dim, kernel_dim,
                                         padding=self.padding, activation=self.conv_activation))
            self.model.add(layers.MaxPooling2D())

        if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.dense_layer_dimension, activation=self.dense_activation))
        self.model.add(layers.Dense(self.num_classes))

    def compile_model(self):
        print("Compiling model")
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                               from_logits=True), metrics=["accuracy"])

    def reset_model_optimizer(self):
        """Reset tensorflow model, optimizer and history by overwriting old instances with new initialization.

        Should only be called after a copy of CNNModel class was created.
        """
        self.model = keras.Sequential()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        self.history = None

    def build_compile(self):
        """Build and compile CNN model."""
        self.build_model()
        self.compile_model()

    def print_summary(self):
        print("Model summary:")
        self.model.summary()

    def train_model(self, train_ds: tf.data.Dataset,
                    val_ds: tf.data.Dataset) -> tf.keras.callbacks.History:
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)
        return self.history

    def test_model(self, test_ds: tf.data.Dataset):
        test_loss, test_acc = self.model.evaluate(x=test_ds)

        print('\nTest accuracy:', test_acc)

    def get_history(self) -> tf.keras.callbacks.History:
        return self.history

    def save_model(self):
        self.model.save(filepath=self.model_path, save_format="tf")

    def load_model(self):
        self.model = tf.keras.models.load_model(filepath=self.model_path)
