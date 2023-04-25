from ppml_datasets import MnistDataset
from model import CNNModel
from util import visualize_training


def train_model():

    mnist = MnistDataset([24, 24, 3], builds_ds_info=False, train_val_test_split=(0.75, 0.20, 0.05))
    mnist.load_dataset()
    mnist.prepare_datasets()

    epochs: int = 40

    cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    cnn_model.build_compile()
    cnn_model.print_summary()

    cnn_model.train_model(mnist.ds_train, mnist.ds_test, epochs=epochs)

    history = cnn_model.get_history()

    visualize_training(history=history, epochs=epochs, img_name="2_cnn.png")

    cnn_model.save_model()


def load_and_test_model():

    mnist = MnistDataset([24, 24, 3], builds_ds_info=False, train_val_test_split=(0.75, 0.20, 0.05))
    mnist.load_dataset()
    mnist.prepare_datasets()

    cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    cnn_model.load_model()
    cnn_model.print_summary()

    cnn_model.test_model(mnist.ds_test)


def main():
    train_model()
    load_and_test_model()


if __name__ == "__main__":
    main()
