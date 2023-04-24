from ppml_datasets import MnistDataset
from model import CNNModel


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


def main():
    train_model()
    load_model()


if __name__ == "__main__":
    main()
