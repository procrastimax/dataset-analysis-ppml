from attacks import AmiaAttack
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.utils import visualize_training, check_create_folder
from ppml_datasets import MnistDataset, FashionMnistDataset, Cifar10Dataset
from util import pickle_object
from cnn_small_model import CNNModel
import pandas as pd

from typing import Optional, Any, Dict, Tuple, List
import os
import sys
import argparse


epochs: int = 500
batch: int = 256
dropout: float = 0.0
learning_rate: float = 0.02
momentum: float = 0.9
weight_decay: Optional[float] = 0.0005
model_input_shape: Tuple[int, int, int] = [32, 32, 3]

data_path: str = "data"
model_path: str = "models"
result_path: str = "results"


def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="Dataset Analysis for Privacy-Preserving-Machine-Learning",
        description="A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.")

    parser.add_argument("-d", "--datasets", nargs="+", required=True, type=str, choices=["mnist", "fmnist", "cifar10"], help="Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here.")
    parser.add_argument("-r", "--run-number", required=True, type=int, help="The run number to be used for training models, loading or saving results.", metavar="R")
    parser.add_argument("-s", "--shadow-model-number", required=False, default=16, type=int, help="The number of shadow models to be trained if '--train-shadow-models' is set.", metavar="N")
    parser.add_argument("e-train-shadow-models", action='store_true', help="If this flag is set, shadow models are trained on 50%% of the given datasets training samples, the other 50%% are used for validating these shadow models. This can be seen as Step 1 in the analysis pipeline.")
    parser.add_argument("--train-single-model", action="store_true", help="If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number.")
    parser.add_argument("--load-test-single-model", action="store_true", help="If this flag is set, a single model is loaded based on run number and dataset name. Then predictions are run on the test and train dataset.")
    parser.add_argument("--run-amia-attack", action="store_true", help="If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved. This can be seen as Step 2 in the analysis pipeline.")
    parser.add_argument("--generate-results", action="store_true", help="If this flag is set, all saved results are compiled and compared with each other, allowing dataset comparison. This can be seen as Step 3 in the analysis pipeline.")
    parser.add_argument("--force-model-retrain", action="store_true", help="If this flag is set, the shadow models, even if they already exist.")
    parser.add_argument("--force-stat-recalculation", action="store_true", help="If this flag is set, the statistics are recalucated on the shadow models.")

    args = parser.parse_args()
    arg_dict: Dict[str, Any] = vars(args)
    return arg_dict


def main():

    args = parse_arguments()

    list_of_ds: List[str] = args["datasets"]
    run_number: int = args["run_number"]
    num_shadow_models: int = args["shadow_model_number"]
    is_training_shadow_models: bool = args["train_shadow_models"]
    is_training_single_model: bool = args["train_single_model"]
    is_load_test_single_model: bool = args["load_test_single_model"]
    is_running_amia_attack: bool = args["run_amia_attack"]
    is_generating_results: bool = args["generate_results"]
    force_model_retraining: bool = args["force_model_retrain"]
    force_stat_recalculation: bool = args["force_stat_recalculation"]

    for ds_name in list_of_ds:
        ds = get_dataset(ds_name)
        ds.load_dataset()
        ds.prepare_datasets()

        num_classes: int = ds.get_number_of_classes()
        model_save_path: str = os.path.join(model_path, str(run_number), ds.dataset_name)
        model = load_model(model_save_path, num_of_classes=num_classes)

        shadow_model_save_path: str = os.path.join(model_path, "shadow_models", str(run_number), ds.dataset_name)
        amia_result_path = os.path.join(result_path, str(run_number))

        if is_training_single_model:
            print("---------------------")
            print("Training single model")
            print("---------------------")
            train_model(ds=ds, model=model, run_number=run_number)

        if is_load_test_single_model:
            print("---------------------")
            print("Loading and testing single model")
            print("---------------------")
            load_and_test_model(ds, model, run_number)

        if is_training_shadow_models:
            print("---------------------")
            print("Training shadow models")
            print("---------------------")
            train_shadow_models(ds, model, run_number,
                                num_shadow_models=num_shadow_models,
                                shadow_model_save_path=shadow_model_save_path,
                                amia_result_path=amia_result_path,
                                force_retrain=force_model_retraining,
                                force_stat_recalculation=force_stat_recalculation)

        if is_running_amia_attack:
            print("---------------------")
            print("Running AMIA attack on shadow models")
            print("---------------------")
            run_amia_attack(ds, model, run_number,
                            num_shadow_models=num_shadow_models,
                            shadow_model_save_path=shadow_model_save_path,
                            amia_result_path=amia_result_path)

        if is_generating_results:
            print("---------------------")
            print("Compiling attack results")
            print("---------------------")


def load_model(model_path: str, num_of_classes: int):
    model = CNNModel(img_height=32, img_width=32, color_channels=3,
                     num_classes=num_of_classes,
                     batch_size=batch,
                     model_path=model_path,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     momentum=momentum,
                     patience=15,
                     use_early_stopping=True,
                     weight_decay=weight_decay)
    return model


def get_dataset(ds_name: str) -> AbstractDataset:
    if ds_name == "mnist":
        return MnistDataset(model_img_shape=model_input_shape, builds_ds_info=False, batch_size=batch, augment_train=False)
    elif ds_name == "fmnist":
        return FashionMnistDataset(model_img_shape=model_input_shape, builds_ds_info=False, batch_size=batch, augment_train=False)
    elif ds_name == "cifar10":
        return Cifar10Dataset(model_img_shape=model_input_shape, builds_ds_info=False, batch_size=batch, augment_train=False)
    else:
        print(f"The requested: {ds_name} dataset does not exist or is not implemented!")
        sys.exit(1)


def train_model(ds: AbstractDataset, model: CNNModel, run_number: int):
    model.build_compile()
    model.print_summary()
    model.train_model_from_ds(train_ds=ds.ds_train, val_ds=ds.ds_test)
    model.save_model()
    history = model.get_history()

    history_fig_filename: str = os.path.join(result_path, str(run_number), "single-model-train", f"{ds.dataset_name}_model_train_history.png")
    check_create_folder(os.path.dirname(history_fig_filename))
    visualize_training(history=history, img_name=history_fig_filename)

    history_save_path: str = os.path.join(model_path, str(run_number), ds.dataset_name, "history")
    check_create_folder(os.path.dirname(history_save_path))
    pickle_object(history_save_path, history)


def load_and_test_model(ds: AbstractDataset, model: CNNModel, run_number: int):
    model.load_model()
    model.print_summary()
    test_df = pd.DataFrame(columns=["type", "accuracy", "loss"])
    print("testing train DS:")
    train_loss, train_acc = model.test_model(ds.ds_train)
    print("testing test DS:")
    test_loss, test_acc = model.test_model(ds.ds_test)
    test_df.loc[0] = ["train", train_acc, train_loss]
    test_df.loc[1] = ["test", test_acc, test_loss]

    result_df_filename = os.path.join(result_path, str(run_number), "single-model-train", f"{ds.dataset_name}_model_predict_results.csv")
    check_create_folder(os.path.dirname(result_df_filename))
    print(f"Saving model test predictions to csv file: {result_df_filename}")
    test_df.to_csv(path_or_buf=result_df_filename, sep="\t", index=False, header=True)


def train_shadow_models(ds: AbstractDataset, model: CNNModel, run_number: int, num_shadow_models: int, shadow_model_save_path: str, amia_result_path: str, force_retrain: bool, force_stat_recalculation: bool):
    amia = AmiaAttack(model=model,
                      ds=ds,
                      num_shadow_models=num_shadow_models,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=amia_result_path,
                      run_name=str(run_number))
    amia.train_load_shadow_models(force_retraning=force_retrain, force_recalculation=force_stat_recalculation)


def run_amia_attack(ds: AbstractDataset, model: CNNModel, run_number: int, num_shadow_models: int, shadow_model_save_path: str, amia_result_path: str):
    amia = AmiaAttack(model=model,
                      ds=ds,
                      num_shadow_models=num_shadow_models,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=amia_result_path,
                      run_name=str(run_number))
    amia.train_load_shadow_models()
    amia.attack_shadow_models_mia()


if __name__ == "__main__":
    main()
