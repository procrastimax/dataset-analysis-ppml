from typing import Optional
import os
from attacks import AmiaAttack
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import visualize_training
from ppml_datasets import MnistDataset, FashionMnistDataset

from cnn_small_model import CNNModel


epochs: int = 500
batch: int = 256
dropout: float = 0.0
learning_rate: float = 0.02
momentum: float = 0.9
weight_decay: Optional[float] = 0.0005


data_path: str = "data"
model_path: str = "models"
result_path: str = "results"


def main():

    run_number: int = 3

    # augment_train is False, since it is built into the model
    ds = FashionMnistDataset([32, 32, 3], builds_ds_info=False, batch_size=batch, augment_train=False)
    ds.load_dataset()
    ds.prepare_datasets()

    num_classes: int = ds.get_number_of_classes()

    model_save_path: str = os.path.join(model_path, str(run_number))
    model = CNNModel(img_height=32, img_width=32, color_channels=3,
                     num_classes=num_classes,
                     batch_size=batch,
                     model_path=model_save_path,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     momentum=momentum,
                     patience=15,
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
                      num_shadow_models=16,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=result_path,
                      run_name=str(run_number))

    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()

    # amia.load_saved_values()
    amia.calculate_tpr_at_fixed_fpr()
    amia.save_all_in_one_roc_curve()


if __name__ == "__main__":
    main()
