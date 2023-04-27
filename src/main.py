from ppml_datasets import MnistDataset
from model import CNNModel
from util import visualize_training
from mia_attack import MiaAttack

from typing import Tuple


epochs: int = 50
train_val_test_split: Tuple[float, float, float] = (0.75, 0.20, 0.05)
batch: int = 30

mnist = MnistDataset([24, 24, 3], builds_ds_info=False, train_val_test_split=train_val_test_split, batch_size=batch, augment_train=False, percentage_loaded_data=5)
cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10, batch_size=batch, dropout=False, model_path="data/models/cnn_no_dropout_model")


def train_model():

    mnist.load_dataset()
    mnist.prepare_datasets()

    cnn_model.build_compile()
    cnn_model.print_summary()

    cnn_model.train_model(mnist.ds_train, mnist.ds_test, epochs=epochs)

    history = cnn_model.get_history()

    visualize_training(history=history, epochs=epochs, img_name="4_cnn.png")

    cnn_model.save_model()


def load_and_test_model():

    mnist.load_dataset()
    mnist.prepare_datasets()

    cnn_model.load_model()
    cnn_model.print_summary()

    cnn_model.test_model(mnist.ds_test)


def run_attack():
    mnist.load_dataset()
    mnist.prepare_datasets()

    cnn_model.load_model()

    mia = MiaAttack(model=cnn_model, dataset=mnist, num_classes=10)
    mia.initialize_data()
    # mia.run_mia_attack()
    mia.calc_membership_probability(plot_training_samples=True)


def main():
    # train_model()
    # load_and_test_model()

    run_attack()


if __name__ == "__main__":
    main()
