from ppml_datasets import Cifar10Dataset
from model import CNNModel
from util import visualize_training
from attacks import AmiaAttack

import os
from typing import Optional

epochs: int = 100
batch: int = 32
dropout: Optional[float] = 0.05

run_number: int = 1
model_name: str = f"r{run_number}_cifar10_e{epochs}"

data_path: str = "data"
model_path: str = "model"
model_save_path: str = os.path.join(data_path, model_path, model_name)

# ds = MnistDataset([27, 27, 3], builds_ds_info=False, batch_size=batch)
ds = Cifar10Dataset([27, 27, 3], builds_ds_info=False, batch_size=batch)

cnn_model = CNNModel(img_height=27, img_width=27, color_channels=3, num_classes=10, batch_size=batch, dropout=dropout, model_path=model_save_path, epochs=epochs, learning_rate=0.1, momentum=0.99, use_l2=True)


def train_model():

    ds.set_augmentation_parameter(random_flip="horizontal", random_rotation=None,
                                  random_zoom=None, random_brightness=None,
                                  random_translation_width=0.1, random_translation_height=0.1)
    ds.load_dataset()
    ds.split_val_from_train(0.3)
    ds.prepare_datasets()

    cnn_model.build_compile()
    cnn_model.print_summary()

    cnn_model.train_model(train_ds=ds.ds_train, val_ds=ds.ds_val)

    history = cnn_model.get_history()

    visualize_training(history=history, img_name=f"{model_name}_cnn_train.png")

    cnn_model.save_model()


def load_and_test_model():

    ds.load_dataset()
    ds.prepare_datasets()

    cnn_model.load_model()
    cnn_model.print_summary()

    cnn_model.test_model(ds.ds_test)


def run_amia_attack():
    ds.load_dataset()
    ds.prepare_datasets()
    cnn_model.load_model()

    amia = AmiaAttack(model=cnn_model, dataset=ds, num_classes=10, num_shadow_models=1)
    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()


def main():
    train_model()
    # load_and_test_model()

    # run_amia_attack()


if __name__ == "__main__":
    main()
