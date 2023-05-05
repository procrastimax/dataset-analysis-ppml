from ppml_datasets import MnistDataset
from ppml_datasets.utils import visualize_training
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from model import CNNModel
from attacks import AmiaAttack

import os
from typing import Optional

epochs: int = 5
batch: int = 256
dropout: float = 0.4
learning_rate: float = 0.001
weight_decay: Optional[float] = 0.01


data_path: str = "data"
model_path: str = "models"
result_path: str = "results"


def main():

    run_number: int = 1

    # augment_train is False, since it is built into the model
    mnist = MnistDataset([32, 32, 3], builds_ds_info=False, batch_size=batch, augment_train=False)
    mnist.load_dataset()
    mnist.prepare_datasets()

    ds = mnist
    num_classes: int = ds.get_number_of_classes()

    model_save_path: str = os.path.join(model_path, str(run_number))
    model = CNNModel(img_height=32, img_width=32, color_channels=3,
                     num_classes=num_classes,
                     batch_size=batch,
                     dropout=dropout,
                     model_path=model_save_path,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     dense_layer_dimension=512,
                     patience=30,
                     use_early_stopping=True,
                     weight_decay=weight_decay)

    # train_model(ds, model)
    # load_and_test_model(ds, model)
    run_amia_attack(ds, model, run_number, result_path)


def train_model(ds: AbstractDataset, model: CNNModel):

    model.build_compile()
    model.print_summary()

    model.train_model_from_ds(train_ds=ds.ds_train, val_ds=ds.ds_test)

    history = model.get_history()

    visualize_training(history=history, img_name=f"{ds.dataset_name}_cnn_train.png")

    model.save_model()


def load_and_test_model(ds: AbstractDataset, model: CNNModel):

    model.load_model()
    model.print_summary()

    print("testing train DS:")
    model.test_model(ds.ds_train)

    print("testing test DS:")
    model.test_model(ds.ds_test)


def run_amia_attack(ds: AbstractDataset, model: CNNModel, run_number: int, result_path: int):

    shadow_model_save_path: str = os.path.join(model_path, "shadow_models")

    amia = AmiaAttack(model=model,
                      ds=ds,
                      num_shadow_models=1,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=result_path,
                      run_name=str(run_number))
    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()


if __name__ == "__main__":
    main()
