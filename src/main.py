from ppml_datasets import Cifar10Dataset
from model import CNNModel
from util import visualize_training
from attacks import AmiaAttack

import os
from typing import Optional

epochs: int = 500
batch: int = 256
dropout: float = 0.2
learning_rate: float = 0.1
weight_decay: Optional[float] = 0.0005

run_number: int = 1
model_name: str = f"r{run_number}_cifar10_e{epochs}_lr{learning_rate}_wd{weight_decay}"

data_path: str = "data"
model_path: str = "model"
model_save_path: str = os.path.join(data_path, model_path, model_name)

# ds = MnistDataset([27, 27, 3], builds_ds_info=False, batch_size=batch)
ds = Cifar10Dataset([32, 32, 3], builds_ds_info=False, batch_size=batch, augment_train=False)

cnn_model = CNNModel(img_height=32, img_width=32, color_channels=3, num_classes=10,
                     batch_size=batch,
                     dropout=dropout,
                     model_path=model_save_path,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     l2_regularization=None,
                     filter_dim_list=[32, 64, 128],
                     kernel_dim_list=[3, 3, 3],
                     dense_layer_dimension=128,
                     patience=100,
                     use_early_stopping=True,
                     weight_decay=weight_decay)


def train_model():

    ds.load_dataset()
    ds.prepare_datasets()

    cnn_model.build_compile()
    cnn_model.print_summary()

    cnn_model.train_model_from_ds(train_ds=ds.ds_train, val_ds=ds.ds_test)

    history = cnn_model.get_history()

    visualize_training(history=history, img_name=f"{model_name}_cnn_train.png")

    cnn_model.save_model()


def load_and_test_model():

    ds.load_dataset()
    ds.prepare_datasets()

    cnn_model.load_model()
    cnn_model.print_summary()

    print("testing train DS:")
    cnn_model.test_model(ds.ds_train)

    print("testing test DS:")
    cnn_model.test_model(ds.ds_test)


def run_amia_attack():
    ds.augment_train = False
    ds.load_dataset()
    ds.prepare_datasets()

    shadow_model_save_path: str = os.path.join(data_path, model_path, "shadow_models", "cifar10", str(run_number))

    amia = AmiaAttack(model=cnn_model, dataset=ds, num_classes=10,
                      num_shadow_models=10, shadow_model_dir=shadow_model_save_path,
                      run_name=str(run_number))
    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()


def main():
    train_model()
    load_and_test_model()
    # run_amia_attack()


if __name__ == "__main__":
    main()
