import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List


class CNNModel():
    def __init__(self, img_height: int, img_width: int, color_channels: int, num_classes: int,
                 conv_layer_filter_dim_list: List[int] = [16, 32, 64],
                 conv_layer_kernel_dim_list: List[int] = [3, 3, 3],
                 dense_layer_dimension: int = 128,
                 optimizer="adam"):
        """Initialize the model."""
        self.img_height = img_width
        self.img_width = img_width
        self.color_channels = color_channels

        # sequential list of convolutional filter dimensions
        self.filter_dim_list = conv_layer_filter_dim_list
        # sequential list of convolutional kernel dimensions
        self.kernel_dim_list = conv_layer_kernel_dim_list

        self.dense_layer_dimension = dense_layer_dimension

        self.num_classes = num_classes

        self.padding: str = "same"
        self.conv_activation: str = "relu"
        self.dense_activation: str = "relu"

        self.optimizer = optimizer

        self.model: keras.Sequential = keras.Sequential()

        self.history: tf.keras.callbacks.History | None = None

    def build_model(self):

        self.model.add(layers.Rescaling(
            1./255, input_shape=(self.img_height, self.img_width, self.color_channels)))

        for (i, (filter_dim, kernel_dim)) in enumerate(zip(self.filter_dim_list,
                                                           self.kernel_dim_list)):
            self.model.add(layers.Conv2D(filter_dim, kernel_dim,
                           padding=self.padding, activation=self.conv_activation))
            self.model.add(layers.MaxPooling2D())

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.dense_layer_dimension, activation=self.dense_activation))
        self.model.add(layers.Dense(self.num_classes))

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                               from_logits=True), metrics=["accuracy"])

    def build_compile(self):
        self.build_model()
        self.compile_model()

    def print_summary(self):
        self.model.summary()

    def train_model(self, train_ds: tf.data.Dataset,
                    val_ds: tf.data.Dataset,
                    epochs: int) -> tf.keras.callbacks.History:
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return self.history

    def get_history(self) -> tf.keras.callbacks.History:
        return self.history
