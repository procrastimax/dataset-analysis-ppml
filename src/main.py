import argparse
import sys
import pandas as pd
import os
from typing import Optional, Any, Dict, Tuple, List
from analyser import AttackAnalyser, UtilityAnalyser
from attacks import AmiaAttack
from util import save_dataframe, plot_histogram


from ppml_datasets.utils import check_create_folder
from util import compute_delta, compute_privacy, compute_noise
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.builder import build_dataset
import tensorflow as tf

import gc
import json

from model import CNNModel, Model, PrivateCNNModel

epochs: int = 20
batch: int = 200
learning_rate: float = 0.001
use_ema: bool = False
ema_momentum: Optional[float] = None  # default value could be 0.99
weight_decay: Optional[float] = None  # default value could be: 0.001

model_input_shape: Tuple[int, int, int] = [32, 32, 3]
random_seed: int = 42
tf.random.set_seed(random_seed)

# Private Training Related Parameter
l2_norm_clip: float = 1.0
num_microbatches: int = batch


data_path: str = "data"
model_path: str = "models"
result_path: str = "results"
ds_info_path: str = "ds-info"


def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="Dataset Analysis for Privacy-Preserving-Machine-Learning",
        description="A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.")

    parser.add_argument("-d", "--datasets", nargs="+", required=False, type=str,
                        help="Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here. Available datasets are: mnist, fmnist, cifar10, cifar100, svhn, emnist-(large|medium|letters|digits|mnist)-(unbalanced|balanced). With modifications _cX (class size), _i[L/N]Y (imbalance), _nX (number of classes), _gray.")
    parser.add_argument("-m", "--model", required=False, type=str, choices=["cnn", "private_cnn"],
                        help="Specify which model should be used for training/ attacking. Only one can be selected!")
    parser.add_argument("-r", "--run-number", required=False, type=int,
                        help="The run number to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results.", metavar="R")
    parser.add_argument("-n", "--run-name", required=False, type=str, default="default",
                        help="The run name to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results. The naming hierarchy here is: model_name/run_name/run_number.", metavar="N")
    parser.add_argument("-s", "--shadow-model-number", required=False, default=16, type=int,
                        help="The number of shadow models to be trained if '--train-shadow-models' is set.", metavar="N")
    parser.add_argument("-tm", "--train-model", action="store_true",
                        help="If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number.")
    parser.add_argument("--epochs", type=int,
                        help="The number of epochs the model should be trained on.")
    parser.add_argument("-l", "--learning-rate", type=float,
                        help="The learning rate used for training models.")
    parser.add_argument("-wd", "--weight-decay", type=float,
                        help="The weight decay used in the Adam optimizer.")
    parser.add_argument("-ema", "--momentum", type=float,
                        help="Momentum value used for Adam's EMA when training the models. If set, EMA in Adam is activated.")
    parser.add_argument("-c", "--l2-norm-clip", type=float,
                        help="The L2 norm clip value set for private training models.")
    parser.add_argument("-b", "--microbatches", type=int,
                        help="Number of microbatches used for private training.")
    parser.add_argument("--batch-size", type=int,
                        help="Size of batch used for training.")
    parser.add_argument("-em", "--evaluate-model", action="store_true",
                        help="If this flag is set, a single model is loaded based on run number, run name, model name and dataset name. Then predictions are run on the test and train dataset to evaluate the model.")
    parser.add_argument("--run-amia-attack", action="store_true",
                        help="If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved.")
    parser.add_argument("--generate-results", action="store_true",
                        help="If this flag is set, all saved results are compiled and compared with each other, allowing dataset comparison.")
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
    parser.add_argument("-e", "--epsilon", type=float, default=1.0,
                        help="The desired epsilon value for DP-SGD learning. Can be any value: 0.1, 1, 10, ...")
    parser.add_argument("-p", "--generate-privacy-report",
                        help="Dont train/load anything, just generate a privacy report for the given values.",
                        action="store_true")
    parser.add_argument("-ce", "--compile-evaluation",
                        help="If this flag is set, the program compiles all single model evaluations from different run numbers to a single file.",
                        action="store_true")

    args = parser.parse_args()
    arg_dict: Dict[str, Any] = vars(args)
    return arg_dict


def main():

    args = parse_arguments()

    list_of_ds: List[str] = args["datasets"]
    run_number: int = args["run_number"]
    run_name: str = args["run_name"]
    model_name: str = args["model"]
    num_shadow_models: int = args["shadow_model_number"]
    is_training_model: bool = args["train_model"]
    is_evaluate_model: bool = args["evaluate_model"]
    is_running_amia_attack: bool = args["run_amia_attack"]
    is_generating_results: bool = args["generate_results"]
    force_model_retraining: bool = args["force_model_retrain"]
    force_stat_recalculation: bool = args["force_stat_recalculation"]
    is_generating_ds_info: bool = args["generate_ds_info"]
    is_including_mia: bool = args["include_mia"]
    is_forcing_ds_info_regeneration: bool = args["force_ds_info_regeneration"]
    is_generating_privacy_report: bool = args["generate_privacy_report"]
    is_compiling_model_evaluation: bool = args["compile_evaluation"]
    privacy_epsilon: float = args["epsilon"]

    arg_momentum: float = args["momentum"]
    arg_learning_rate: float = args["learning_rate"]
    arg_weight_decay: float = args["weight_decay"]

    arg_l2_clip_norm: float = args["l2_norm_clip"]
    arg_microbatches: int = args["microbatches"]
    arg_epochs: int = args["epochs"]
    arg_batch: int = args["batch_size"]

    if arg_momentum is not None:
        global ema_momentum
        ema_momentum = arg_momentum

        global use_ema
        use_ema = True

    if arg_weight_decay is not None:
        global weight_decay
        weight_decay = arg_weight_decay

    if arg_learning_rate is not None:
        global learning_rate
        learning_rate = arg_learning_rate

    if arg_l2_clip_norm is not None:
        global l2_norm_clip
        l2_norm_clip = arg_l2_clip_norm

    if arg_epochs is not None:
        global epochs
        epochs = arg_epochs

    if arg_batch is not None:
        global batch
        global num_microbatches
        batch = arg_batch
        num_microbatches = batch

    if arg_microbatches is not None:
        num_microbatches = arg_microbatches

    if is_generating_privacy_report:
        print("Calculating privacy statement for given parameter...")

        used_microbatching = True
        if num_microbatches <= 1:
            used_microbatching = False

        num_train_samples = 50000
        delta = compute_delta(num_train_samples)
        # assuming MNIST case here
        noise = compute_noise(num_train_samples=num_train_samples,
                              batch_size=batch,
                              target_epsilon=privacy_epsilon,
                              epochs=epochs,
                              delta=delta)

        priv_report = compute_privacy(num_train_samples,
                                      batch,
                                      noise,
                                      epochs,
                                      delta,
                                      used_microbatching=used_microbatching)
        print(priv_report)

        sys.exit(0)

    loaded_ds_list: List[AbstractDataset] = []

    ds_info_df = pd.DataFrame()

    if model_name is None:
        print("No model was specified! Please provide a valid model name!")
        sys.exit(1)

    amia_result_path = os.path.join(result_path, model_name, run_name, str(run_number))
    ds_info_path = "ds-info"

    if is_training_model or is_evaluate_model or is_running_amia_attack:
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

    if list_of_ds is not None:
        list_of_ds.sort()  # sort ds name list to create deterministic filenames

    single_model_test_df = None
    model = None

    print("=========================================")
    print("=========================================")
    print("=========================================")

    if list_of_ds is not None:
        for ds_name in list_of_ds:
            # create folder for ds-info
            # we need this since we save some more information on datasets
            ds_info_path_specific = os.path.join("ds-info", ds_name)
            check_create_folder(ds_info_path_specific)

            ds = build_dataset(ds_name, batch_size=batch, model_input_shape=model_input_shape)

            # generate ds_info before preprocessing dataset
            if is_generating_ds_info:
                print("---------------------")
                print("Generating Dataset Info")
                print("---------------------")
                ds_info_df = generate_ds_info(ds_info_path=ds_info_path_specific,
                                              ds=ds,
                                              ds_info_df=ds_info_df,
                                              force_ds_info_regen=is_forcing_ds_info_regeneration)
            ds.prepare_datasets()
            loaded_ds_list.append(ds)

            if is_training_model or is_evaluate_model or is_running_amia_attack:
                model_save_path: str = os.path.join(
                    model_path, model_name, run_name, str(run_number), ds.dataset_name)
                check_create_folder(model_save_path)
                model_save_file: str = os.path.join(model_save_path, f"{ds.dataset_name}.keras")
                model = load_model(model_path=model_save_file,
                                   model_name=model_name,
                                   num_classes=ds.num_classes)

                # set values for private training
                if type(model) is PrivateCNNModel:
                    print(
                        f"Setting private training parameter epsilon: {privacy_epsilon}, l2_norm_clip: {l2_norm_clip}, num_microbatches: {num_microbatches}")
                    num_train_samples = int(len(ds.get_train_ds_as_numpy()[0]))
                    model.set_privacy_parameter(epsilon=privacy_epsilon,
                                                num_train_samples=num_train_samples,
                                                l2_norm_clip=l2_norm_clip,
                                                num_microbatches=num_microbatches)

            if is_training_model:
                print("---------------------")
                print("Training single model")
                print("---------------------")
                train_model(ds=ds, model=model, run_name=run_name, run_number=run_number)

            if is_evaluate_model:
                print("---------------------")
                print("Loading and evaluate model")
                print("---------------------")
                result_df = load_and_test_model(ds, model)
                if single_model_test_df is None:
                    single_model_test_df = result_df
                else:
                    single_model_test_df = pd.concat([single_model_test_df, result_df])

            if is_running_amia_attack:
                print("---------------------")
                print("Running AMIA attack (train shadow models, calc statistics, run attack)")
                print("---------------------")
                shadow_model_save_path: str = os.path.join(
                    model_path, model_name, run_name, str(run_number), "shadow_models", ds.dataset_name)
                check_create_folder(shadow_model_save_path)

                # make sure that the num_microbatches var is not set when training non-private models
                if type(model) is not PrivateCNNModel:
                    num_microbatches = None

                run_amia_attack(ds=ds,
                                model=model,
                                num_shadow_models=num_shadow_models,
                                shadow_model_save_path=shadow_model_save_path,
                                amia_result_path=amia_result_path,
                                force_retrain=force_model_retraining,
                                force_stat_recalculation=force_stat_recalculation,
                                include_mia=is_including_mia,
                                num_microbatches=num_microbatches)

    if is_generating_results:
        print("---------------------")
        print("Compiling attack results")
        print("---------------------")
        analyser = AttackAnalyser(ds_list=loaded_ds_list,
                                  model_name=model.model_name,
                                  run_name=run_name,
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

    if is_evaluate_model:
        # save result_df with model's accuracy, loss, etc. gathered as csv
        result_df_filename = os.path.join(result_path, model.model_name, run_name, str(
            run_number), "single-model-train", f'{"-".join(list_of_ds)}_model_predict_results.csv')
        check_create_folder(os.path.dirname(result_df_filename))
        save_dataframe(df=single_model_test_df, filename=result_df_filename, use_index=False)

    if model is not None:
        # save run parameter as json
        run_params = {
            "epochs": epochs,
            "batch": batch,
            "learning_rate": learning_rate,
            "ema_momentum": ema_momentum,
            "weight_decay": weight_decay,
            "shadow_models": num_shadow_models,
            "privacy_epsilon": privacy_epsilon,
            "l2_norm_clip": l2_norm_clip,
            "num_microbatches": num_microbatches}

        param_filepath = os.path.join(result_path, model.model_name, run_name,
                                      str(run_number))
        check_create_folder(param_filepath)
        param_filepath = os.path.join(param_filepath, "parameter.csv")
        print(f"Saving program parameter to: {param_filepath}")
        with open(param_filepath, "w") as f:
            json.dump(run_params, f)

    if is_compiling_model_evaluation:
        print("---------------------")
        print("Compiling model evaluation")
        print("---------------------")
        cwd = os.getcwd()
        res_path = os.path.join(cwd, result_path)
        analyser = UtilityAnalyser(result_path=res_path,
                                   run_name=run_name,
                                   model_name=model_name)
        analyser.analyse_utility()


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


def load_model(model_path: str, model_name: str, num_classes: int) -> Model:
    model = None

    model_height = model_input_shape[0]
    model_width = model_input_shape[1]
    model_color_space = model_input_shape[2]

    if model_name == "cnn":
        model = CNNModel(model_name="cnn",
                         img_height=model_height,
                         img_width=model_width,
                         color_channels=model_color_space,
                         random_seed=random_seed,
                         num_classes=num_classes,
                         batch_size=batch,
                         model_path=model_path,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         ema_momentum=ema_momentum,
                         weight_decay=weight_decay)
    elif model_name == "private_cnn":
        model = PrivateCNNModel(model_name="private_cnn",
                                img_height=model_height,
                                img_width=model_width,
                                color_channels=model_color_space,
                                random_seed=random_seed,
                                num_classes=num_classes,
                                batch_size=batch,
                                model_path=model_path,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                ema_momentum=ema_momentum,
                                weight_decay=weight_decay)
    return model


def train_model(ds: AbstractDataset, model: Model, run_name: str, run_number: int):
    model.build_compile()
    model.print_summary()

    x, y = ds.convert_ds_to_one_hot_encoding(ds.ds_train, unbatch=True)
    test_x, test_y = ds.convert_ds_to_one_hot_encoding(ds.ds_test, unbatch=True)

    model.train_model_from_numpy(x=x, y=y, val_x=test_x, val_y=test_y)
    model.save_model()

    train_history_folder = os.path.join(
        result_path, model.model_name, run_name, str(run_number), "single-model-train")
    model.save_train_history(folder_name=train_history_folder,
                             image_name=f"{ds.dataset_name}_model_train_history.png")
    # avoid OOM
    tf.keras.backend.clear_session()
    gc.collect()


def load_and_test_model(ds: AbstractDataset, model: Model) -> pd.DataFrame:
    model.load_model()
    model.compile_model()
    model.print_summary()

    x, y = ds.convert_ds_to_one_hot_encoding(ds.ds_train, unbatch=True)
    test_x, test_y = ds.convert_ds_to_one_hot_encoding(ds.ds_test, unbatch=True)

    train_eval_dict = model.test_model(x, y)
    test_eval_dict = model.test_model(test_x, test_y)

    train_eval_dict["type"] = "train"
    test_eval_dict["type"] = "test"

    train_eval_dict["name"] = ds.dataset_name
    test_eval_dict["name"] = ds.dataset_name

    merged_dicts = {}
    for k, v in train_eval_dict.items():
        merged_dicts[k] = [v, test_eval_dict[k]]

    df = pd.DataFrame.from_dict(merged_dicts)
    df = df[['name', 'type', 'accuracy', 'f1-score', 'precision', 'recall', 'loss']]
    return df


def run_amia_attack(ds: AbstractDataset,
                    model: Model,
                    num_shadow_models: int,
                    shadow_model_save_path: str,
                    amia_result_path: str,
                    force_retrain: bool,
                    force_stat_recalculation: bool,
                    include_mia: bool,
                    num_microbatches: Optional[int] = None):
    amia = AmiaAttack(model=model,
                      ds=ds,
                      num_shadow_models=num_shadow_models,
                      shadow_model_dir=shadow_model_save_path,
                      result_path=amia_result_path,
                      include_mia=include_mia)
    amia.train_load_shadow_models(force_retraining=force_retrain,
                                  force_recalculation=force_stat_recalculation,
                                  num_microbatch=num_microbatches)
    amia.attack_shadow_models_amia()


if __name__ == "__main__":
    main()
