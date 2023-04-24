from ppml_datasets import MnistDataset
from model import CNNModel

import tensorflow as tf


def train_model():

    mnist = MnistDataset([24, 24, 3], builds_ds_info=False)
    mnist.load_dataset()
    mnist.prepare_datasets()

    epochs: int = 1

    cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    cnn_model.build_compile()
    cnn_model.print_summary()

    cnn_model.train_model(mnist.ds_train, mnist.ds_test, epochs=epochs)

    cnn_model.test_model(mnist.ds_test)

    cnn_model.save_model()


def load_model():
    cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    cnn_model.load_model()
    cnn_model.print_summary()


def test_dataset():
    mnist = MnistDataset([24, 24, 3], builds_ds_info=False, train_val_test_split=(0.75, 0.20, 0.05))
    mnist.load_dataset()
    mnist.build_ds_info()
    print(f"{mnist.ds_train} - {len(mnist.ds_train)}")
    print(f"{mnist.ds_val} - {len(mnist.ds_val)}")

    mnist.prepare_datasets()

    print(f"{mnist.ds_train} - {len(mnist.ds_train)}")
    print(f"{mnist.ds_val} - {len(mnist.ds_val)}")
    print(f"{mnist.ds_test} - {len(mnist.ds_test)}")

    print(mnist.ds_info)


def main():
    test_dataset()


if __name__ == "__main__":
    main()
