import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List, Optional


class CNNModel():
    def __init__(self, img_height: int, img_width: int, color_channels: int, num_classes: int,
                 conv_layer_filter_dim_list: List[int] = [16, 32, 64],
                 conv_layer_kernel_dim_list: List[int] = [3, 3, 3],
                 dense_layer_dimension: int = 128,
                 batch_size: int = 32,
                 model_path: str = "data/models/cnn_model",
                 dropout: Optional[float] = 0.2,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0):
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

        self.batch_size = batch_size

        self.model: keras.Sequential = keras.Sequential()

        self.history: tf.keras.callbacks.History | None = None

        self.model_path = model_path

        self.dropout: Optional[float] = dropout

        self.learning_rate: float = learning_rate
        self.momentum: float = momentum

        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

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

    def build_compile(self):
        self.build_model()
        self.compile_model()

    def print_summary(self):
        print("Model summary:")
        self.model.summary()

    def train_model(self, train_ds: tf.data.Dataset,
                    val_ds: tf.data.Dataset,
                    epochs: int) -> tf.keras.callbacks.History:
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return self.history

    def test_model(self, test_ds: tf.data.Dataset):
        test_loss, test_acc = self.model.evaluate(x=test_ds, batch_size=self.batch_size)

        print('\nTest accuracy:', test_acc)

    def get_history(self) -> tf.keras.callbacks.History:
        return self.history

    def save_model(self):
        self.model.save(filepath=self.model_path, save_format="tf")

    def load_model(self):
        self.model = tf.keras.models.load_model(filepath=self.model_path)
