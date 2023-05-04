from ppml_datasets import Cifar10Dataset
from ppml_datasets.utils import filter_labels, visualize_training
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from model import CNNModel
from attacks import AmiaAttack

import os
from typing import Optional

epochs: int = 500
batch: int = 256
dropout: float = 0.4
learning_rate: float = 0.001
weight_decay: Optional[float] = 0.01


run_number: int = 1
classes = list(range(10))
num_classes = len(classes)

model_name: str = f"cifar{num_classes}"
model_name: str = f"r{run_number}_{model_name}_e{epochs}_lr{learning_rate}_wd{weight_decay}"

data_path: str = "data"
model_path: str = "model"
model_save_path: str = os.path.join(data_path, model_path, model_name)


def main():

    # mnist = MnistDataset([27, 27, 3], builds_ds_info=False, batch_size=batch, augment_train=False)
    # cifar10 = Cifar10Dataset([32, 32, 3], builds_ds_info=False, batch_size=batch, augment_train=False)

    cifarx = Cifar10Dataset([32, 32, 3], builds_ds_info=False, batch_size=batch, augment_train=False)
    # cifarx.load_dataset(lambda x, y: filter_labels(y, list(range(10))))
    cifarx.load_dataset()

    cifarx.prepare_datasets()

    ds = cifarx

    model = CNNModel(img_height=32, img_width=32, color_channels=3,
                     num_classes=num_classes,
                     batch_size=batch,
                     dropout=dropout,
                     model_path=model_save_path,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     dense_layer_dimension=512,
                     patience=50,
                     use_early_stopping=True,
                     weight_decay=weight_decay)

    train_model(ds, model)
    load_and_test_model(ds, model)
    # run_amia_attack(ds, model)


def train_model(ds: AbstractDataset, model: CNNModel):

    model.build_compile()
    model.print_summary()

    model.train_model_from_ds(train_ds=ds.ds_train, val_ds=ds.ds_test)

    history = model.get_history()

    visualize_training(history=history, img_name=f"{model_name}_cnn_train.png")

    model.save_model()


def load_and_test_model(ds: AbstractDataset, model: CNNModel):

    model.load_model()
    model.print_summary()

    print("testing train DS:")
    model.test_model(ds.ds_train)

    print("testing test DS:")
    model.test_model(ds.ds_test)


def run_amia_attack(ds: AbstractDataset, model: CNNModel):

    shadow_model_save_path: str = os.path.join(data_path, model_path, "shadow_models", "cifar10", str(run_number))

    amia = AmiaAttack(model=model, dataset=ds, num_classes=10,
                      num_shadow_models=10, shadow_model_dir=shadow_model_save_path,
                      run_name=str(run_number))
    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()


if __name__ == "__main__":
    main()
