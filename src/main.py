import argparse
import sys
import pandas as pd
import os
from typing import Optional, Any, Dict, Tuple, List
from analyser import Analyser
from attacks import AmiaAttack
from util import save_dataframe, plot_histogram
from ppml_datasets import MnistDataset, FashionMnistDataset, Cifar10Dataset, Cifar10DatasetGray, MnistDatasetCustomClassSize, FashionMnistDatasetCustomClassSize
from ppml_datasets.utils import check_create_folder
from ppml_datasets.abstract_dataset_handler import AbstractDataset

import json

from model import SmallCNNModel, Model, PrivateSmallCNNModel

epochs: int = 150
batch: int = 256
learning_rate: float = 0.25
momentum: Optional[float] = None
weight_decay: Optional[float] = None
model_input_shape: Tuple[int, int, int] = [32, 32, 3]

shadow_models: int = 16

# Private Training Related Parameter
l2_norm_clip: float = 1.0
num_microbatches: int = 32


data_path: str = "data"
model_path: str = "models"
result_path: str = "results"
ds_info_path: str = "ds-info"


def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="Dataset Analysis for Privacy-Preserving-Machine-Learning",
        description="A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.")

    parser.add_argument("-d", "--datasets", nargs="+", required=False, type=str, choices=["mnist", "mnist_c5000", "fmnist", "fmnist_c5000", "cifar10", "cifar10gray"],
                        help="Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here.")
    parser.add_argument("-m", "--model", required=False, type=str, choices=["small_cnn", "private_small_cnn"],
                        help="Specify which model should be used for training/ attacking. Only one can be selected!")
    parser.add_argument("-r", "--run-number", required=False, type=int,
                        help="The run number to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results.", metavar="R")
    parser.add_argument("-s", "--shadow-model-number", required=False, default=16, type=int,
                        help="The number of shadow models to be trained if '--train-shadow-models' is set.", metavar="N")
    parser.add_argument("--train-single-model", action="store_true",
                        help="If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number.")
    parser.add_argument("--epochs", type=int,
                        help="The number of epochs the model should be trained on.")
    parser.add_argument("-l", "--learning-rate", type=float,
                        help="The learning rate used for training models.")
    parser.add_argument("--momentum", type=float,
                        help="Momentum value used for training the models.")
    parser.add_argument("-c", "--l2-norm-clip", type=float,
                        help="The L2 norm clip value set for private training models.")
    parser.add_argument("-b", "--microbatches", type=int,
                        help="Number of microbatches used for private training.")
    parser.add_argument("--load-test-single-model", action="store_true",
                        help="If this flag is set, a single model is loaded based on run number and dataset name. Then predictions are run on the test and train dataset.")
    parser.add_argument("--run-amia-attack", action="store_true",
                        help="If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved. This can be seen as Step 2 in the analysis pipeline.")
    parser.add_argument("--generate-results", action="store_true",
                        help="If this flag is set, all saved results are compiled and compared with each other, allowing dataset comparison. This can be seen as Step 3 in the analysis pipeline.")
    parser.add_argument("--force-model-retrain", action="store_true",
                        help="If this flag is set, the shadow models, even if they already exist.")
    parser.add_argument("--force-stat-recalculation", action="store_true",
                        help="If this flag is set, the statistics are recalucated on the shadow models.")
    parser.add_argument("--generate-ds-info", action="store_true",
                        help="If this flag is set, dataset infos are generated and saved.")
    parser.add_argument("--force-ds-info-regeneration", action="store_true",
                        help="If this flag is set, the whole ds-info dict is not loaded from a json file but regenerated from scratch.")
    parser.add_argument("--include-mia", action="store_true",
                        help="If this flag is set, then the mia attack is also used during attacking and mia related results/ graphics are produced during result generation.")
    parser.add_argument("-e", "--epsilon", type=float,
                        help="The desired epsilon value for DP-SGD learning. Can be any value: 0, 0.1, 1, 10, None (if not set)")

    args = parser.parse_args()
    arg_dict: Dict[str, Any] = vars(args)
    return arg_dict


def main():

    args = parse_arguments()

    list_of_ds: List[str] = args["datasets"]
    run_number: int = args["run_number"]
    model_name: str = args["model"]
    num_shadow_models: int = args["shadow_model_number"]
    is_training_single_model: bool = args["train_single_model"]
    is_load_test_single_model: bool = args["load_test_single_model"]
    is_running_amia_attack: bool = args["run_amia_attack"]
    is_generating_results: bool = args["generate_results"]
    force_model_retraining: bool = args["force_model_retrain"]
    force_stat_recalculation: bool = args["force_stat_recalculation"]
    is_generating_ds_info: bool = args["generate_ds_info"]
    is_including_mia: bool = args["include_mia"]
    is_forcing_ds_info_regeneration: bool = args["force_ds_info_regeneration"]
    privacy_epsilon: float = args["epsilon"]
    arg_momentum: float = args["momentum"]
    arg_learning_rate: float = args["learning_rate"]
    arg_l2_clip_norm: float = args["l2_norm_clip"]
    arg_microbatches: int = args["microbatches"]
    arg_epochs: int = args["epochs"]

    if arg_momentum is not None:
        global momentum
        momentum = arg_momentum

    if arg_learning_rate is not None:
        global learning_rate
        learning_rate = arg_learning_rate

    if arg_l2_clip_norm is not None:
        global l2_norm_clip
        l2_norm_clip = arg_l2_clip_norm

    if arg_microbatches is not None:
        global num_microbatches
        num_microbatches = arg_microbatches

    if arg_epochs is not None:
        global epochs
        epochs = arg_epochs

    loaded_ds_list: List[AbstractDataset] = []

    ds_info_df = pd.DataFrame()

    amia_result_path = os.path.join(result_path, str(run_number))
    ds_info_path = "ds-info"

    if is_training_single_model or is_load_test_single_model or is_running_amia_attack:
        if run_number is None:
            print("No run number specified! A run number is required when training/ attacking/ testing models!")
            sys.exit(1)

        if model_name is None:
            print("No model specified! A model is required when training/ attacking/ testing models!")
            sys.exit(1)

        if list_of_ds is None:
            print("No datasets specified! A datasets is required when training/ attacking/ testing models!")
            sys.exit(1)

    if is_generating_ds_info and list_of_ds is None:
        print("No datasets specified! A datasets is required when training/ attacking/ testing models!")
        sys.exit(1)

    list_of_ds.sort()  # sort ds name list to create deterministic filenames

    single_model_test_df = None

    for ds_name in list_of_ds:
        ds = get_dataset(ds_name)

        # generate ds_info before preprocessing dataset
        if is_generating_ds_info:
            print("---------------------")
            print("Generating Dataset Info")
            print("---------------------")
            ds_info_path_specific = os.path.join("ds-info", ds.dataset_name)
            check_create_folder(ds_info_path_specific)
            ds_info_df = generate_ds_info(ds_info_path=ds_info_path_specific,
                                          ds=ds,
                                          ds_info_df=ds_info_df,
                                          force_ds_info_regen=is_forcing_ds_info_regeneration)
        ds.prepare_datasets()
        loaded_ds_list.append(ds)

        model = None
        if is_training_single_model or is_load_test_single_model or is_running_amia_attack:
            model_save_path: str = os.path.join(
                model_path, model_name, str(run_number), ds.dataset_name)
            check_create_folder(model_save_path)
            model_save_file: str = os.path.join(model_save_path, f"{ds.dataset_name}.h5")
            model = load_model(model_save_file,
                               model_name=model_name,
                               num_of_classes=ds.num_classes)

            # set values for private training
            if model.is_private_model:
                print(
                    f"Setting private training parameter epsilon: {privacy_epsilon}, l2_norm_clip: {l2_norm_clip}, num_microbatches: {num_microbatches}")
                num_train_samples = int(len(ds.get_train_ds_as_numpy()[0]))
                model.set_privacy_parameter(epsilon=privacy_epsilon,
                                            num_train_samples=num_train_samples,
                                            l2_norm_clip=l2_norm_clip,
                                            num_microbatches=num_microbatches)

        if is_training_single_model:
            print("---------------------")
            print("Training single model")
            print("---------------------")
            train_model(ds=ds, model=model, run_number=run_number)

        if is_load_test_single_model:
            print("---------------------")
            print("Loading and testing single model")
            print("---------------------")
            result_df = load_and_test_model(ds, model, run_number)
            if single_model_test_df is None:
                single_model_test_df = result_df
            else:
                single_model_test_df = pd.concat([single_model_test_df, result_df])

        if is_running_amia_attack:
            print("---------------------")
            print("Running AMIA attack (train shadow models, calc statistics, run attack)")
            print("---------------------")
            shadow_model_save_path: str = os.path.join(
                model_path, str(run_number), "shadow_models", ds.dataset_name)
            check_create_folder(shadow_model_save_path)
            run_amia_attack(ds=ds,
                            model=model,
                            num_shadow_models=num_shadow_models,
                            shadow_model_save_path=shadow_model_save_path,
                            amia_result_path=amia_result_path,
                            force_retrain=force_model_retraining,
                            force_stat_recalculation=force_stat_recalculation,
                            include_mia=is_including_mia)

    if is_generating_results:
        print("---------------------")
        print("Compiling attack results")
        print("---------------------")
        analyser = Analyser(ds_list=loaded_ds_list,
                            run_number=run_number,
                            result_path=result_path,
                            model_path=model_path,
                            num_shadow_models=num_shadow_models,
                            include_mia=is_including_mia)
        analyser.generate_results()

    if is_generating_ds_info:
        print("---------------------")
        print("Saving Dataset Info")
        print("---------------------")
        print(ds_info_df)
        ds_info_df_file = os.path.join(
            ds_info_path, f'dataframe_{"-".join(list_of_ds)}_ds_info.csv')
        save_dataframe(ds_info_df, ds_info_df_file)

    if is_load_test_single_model:
        # save result_df with model's accuracy, loss, etc. gathered as csv
        result_df_filename = os.path.join(result_path, model.model_name, str(
            run_number), "single-model-train", f'{"-".join(list_of_ds)}_model_predict_results.csv')
        check_create_folder(os.path.dirname(result_df_filename))
        save_dataframe(df=single_model_test_df, filename=result_df_filename, use_index=False)

    # save run parameter as json
    run_params = {"epochs": epochs,
                  "batch": batch,
                  "learning_rate": learning_rate,
                  "momentum": momentum,
                  "weight_decay": weight_decay,
                  "shadow_models": shadow_models,
                  "l2_norm_clip": l2_norm_clip,
                  "num_microbatches": num_microbatches}

    param_filepath = os.path.join(result_path, model.model_name, str(run_number), "parameter.csv")
    print(f"Saving program parameter to: {param_filepath}")
    with open(param_filepath, "w") as f:
        json.dump(run_params, f)


def generate_ds_info(ds_info_path: str, ds: AbstractDataset, ds_info_df: pd.DataFrame, force_ds_info_regen: bool) -> pd.DataFrame:
    """Generate dataset info.

    Needs to be run before dataset preprocessing is called.

    Return:
    ------
    pd.Dataframe -> Dataframe to compare different datasets by single-value metrics.

    """
    check_create_folder(ds_info_path)

    ds.build_ds_info(force_regeneration=force_ds_info_regen)

    hist_filename = os.path.join(ds_info_path, "histogram",
                                 f"train_data_hist_{ds.dataset_name}.png")
    hist_filename_mean = os.path.join(ds_info_path, "histogram",
                                      f"mean_train_data_hist_{ds.dataset_name}.png")
    check_create_folder(os.path.dirname(hist_filename))
    # save histogram
    hist, bins = ds.get_data_histogram(use_mean=False)
    plot_histogram(hist, bins, hist_filename, title="Train Data Histogram",
                   xlabel="Pixel Value", ylabel="Probability")

    hist, bins = ds.get_data_histogram(use_mean=True)
    plot_histogram(hist, bins, hist_filename_mean, title="Train Data Histogram (Averaged)",
                   xlabel="Pixel Value", ylabel="Probability")

    ds.save_ds_info_as_json()

    df = ds.get_ds_info_as_df()
    ds_info_df = pd.concat([ds_info_df, df])

    return ds_info_df


def load_model(model_path: str, model_name: str, num_of_classes: int) -> Model:
    model = None
    if model_name == "small_cnn":
        model = SmallCNNModel(img_height=32,
                              img_width=32,
                              color_channels=3,
                              num_classes=num_of_classes,
                              batch_size=batch,
                              model_name="small_cnn",
                              model_path=model_path,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              momentum=momentum,
                              patience=15,
                              use_early_stopping=True,
                              is_private_model=False)
    elif model_name == "private_small_cnn":
        model = PrivateSmallCNNModel(img_height=32,
                                     img_width=32,
                                     color_channels=3,
                                     num_classes=num_of_classes,
                                     batch_size=batch,
                                     model_name="private_small_cnn",
                                     model_path=model_path,
                                     epochs=epochs,
                                     learning_rate=learning_rate,
                                     momentum=momentum,
                                     patience=15,
                                     use_early_stopping=False,
                                     is_private_model=True)
    return model


def get_dataset(ds_name: str) -> AbstractDataset:
    ds = None

    if ds_name == "mnist":
        ds = MnistDataset(model_img_shape=model_input_shape,
                          builds_ds_info=False,
                          batch_size=batch,
                          augment_train=False)

    elif ds_name == "mnist_c5000":
        ds = MnistDatasetCustomClassSize(model_img_shape=model_input_shape,
                                         class_size=5000,
                                         builds_ds_info=False,
                                         batch_size=batch,
                                         augment_train=False)

    elif ds_name == "fmnist":
        ds = FashionMnistDataset(model_img_shape=model_input_shape,
                                 builds_ds_info=False,
                                 batch_size=batch,
                                 augment_train=False)

    elif ds_name == "fmnist_c5000":
        ds = FashionMnistDatasetCustomClassSize(
            model_img_shape=model_input_shape,
            class_size=5000,
            builds_ds_info=False,
            batch_size=batch,
            augment_train=False)

    elif ds_name == "cifar10":
        ds = Cifar10Dataset(model_img_shape=model_input_shape,
                            builds_ds_info=False,
                            batch_size=batch,
                            augment_train=False)

    elif ds_name == "cifar10gray":
        ds = Cifar10DatasetGray(model_img_shape=model_input_shape,
                                builds_ds_info=False,
                                batch_size=batch,
                                augment_train=False)
    else:
        print(f"The requested: {ds_name} dataset does not exist or is not implemented!")
        sys.exit(1)

    ds.ds_info_path = os.path.join(ds_info_path, ds.dataset_name)
    ds.load_dataset()
    return ds


def train_model(ds: AbstractDataset, model: Model, run_number: int):
    model.build_compile()
    model.print_summary()
    model.train_model_from_ds(train_ds=ds.ds_train, val_ds=ds.ds_test)
    model.save_model()

    train_history_folder = os.path.join(
        result_path, model.model_name, str(run_number), "single-model-train")
    model.save_train_history(folder_name=train_history_folder,
                             image_name=f"{ds.dataset_name}_model_train_history.png")


def load_and_test_model(ds: AbstractDataset, model: Model, run_number: int) -> pd.DataFrame:
    model.load_model()
    model.compile_model()
    model.print_summary()

    train_loss, train_acc = model.test_model(ds.ds_train)
    test_loss, test_acc = model.test_model(ds.ds_test)
    data = [[ds.dataset_name, "train", train_acc, train_loss]]
    data.append([ds.dataset_name, "test", test_acc, test_loss])
    test_df = pd.DataFrame(data=data, columns=["dataset", "type", "accuracy", "loss"])
    return test_df


def run_amia_attack(ds: AbstractDataset,
                    model: Model,
                    num_shadow_models: int,
                    shadow_model_save_path: str,
                    amia_result_path: str,
                    force_retrain: bool,
                    force_stat_recalculation: bool,
                    include_mia: bool):
    amia = AmiaAttack(model=model,
                      ds=ds,
                      num_shadow_models=num_shadow_models,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=amia_result_path,
                      include_mia=include_mia)
    amia.train_load_shadow_models(force_retraining=force_retrain,
                                  force_recalculation=force_stat_recalculation)
    amia.attack_shadow_models_amia()


if __name__ == "__main__":
    main()
