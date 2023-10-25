import gc
import json
import os
import sys
from datetime import datetime as dt
from typing import List, Optional, Tuple

import pandas as pd
import tensorflow as tf

from attack_analyser import AttackAnalyser, AttackType
from attacks import AmiaAttack
from model import CNNModel, Model, PrivateCNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.builder import build_dataset
from ppml_datasets.utils import check_create_folder
from settings import RunSettings, create_arg_parse_instance, create_settings_from_args
from util import (
    compute_delta,
    compute_noise,
    compute_privacy,
    get_run_numbers,
    save_dataframe,
)
from utility_analyser import UtilityAnalyser

data_path: str = "data"
model_path: str = "models"
result_path: str = "results"
ds_info_path: str = "ds-info"

# get starttime of program
start_time = dt.now()


def main():
    parser = create_arg_parse_instance()
    args = parser.parse_args()

    settings = create_settings_from_args(args)

    settings.print_values()

    tf.random.set_seed(settings.random_seed)

    if settings.is_generating_privacy_report:
        generate_privacy_report(settings, num_samples=60000)

    loaded_ds_list: List[AbstractDataset] = []

    run_args_parameter_check(settings)

    if settings.datasets is not None:
        settings.datasets.sort(
        )  # sort ds name list to create deterministic filenames

    ds_info_df_all: Optional[pd.DataFrame] = None

    print("=========================================")
    print("=========================================")
    print("=========================================")

    # if no datasets were specified, then load them from paramter.json files for each run from a 'run name' folder
    if settings.datasets is None and (settings.is_train_model
                                      or settings.is_evaluating_model):
        run_result_folder = os.path.join(result_path, settings.model_name,
                                         settings.run_name)

        if settings.analysis_run_numbers is None:
            runs = get_run_numbers(run_result_folder)
        else:
            runs = settings.analysis_run_numbers
        print(f"Found the following runs for {settings.run_name}: {runs}")
        for run in runs:
            parameter_file_path = os.path.join(run_result_folder, str(run),
                                               "parameter_model_train.json")
            with open(parameter_file_path) as parameter_file:
                parameter_dict = json.load(parameter_file)
                # load and set run parameter from parameter.json file
                settings.epochs = parameter_dict["epochs"]
                settings.batch = parameter_dict["batch"]
                settings.learning_rate = parameter_dict["learning_rate"]
                settings.ema_momentum = parameter_dict["ema_momentum"]
                settings.weight_decay = parameter_dict["weight_decay"]
                settings.delta = parameter_dict["delta"]
                settings.l2_norm_clip = parameter_dict["l2_norm_clip"]
                settings.privacy_epsilon = parameter_dict["privacy_epsilon"]
                settings.num_shadow_models = parameter_dict[
                    "num_shadow_models"]
                settings.adam_epsilon = parameter_dict["adam_epsilon"]
                settings.random_seed = parameter_dict["random_seed"]
                settings.datasets = parameter_dict["datasets"]

            if settings.datasets is None:
                print(
                    "Cannot load attack results, since the parameter_model_train.json file does not contain used dataset names for the runs!"
                )
                sys.exit(1)

            print(
                f"Loading the following datasets for run {run}: {settings.datasets}"
            )

            # update settings' run number for this iteration
            settings.run_number = int(run)

            single_model_test_df: Optional[pd.DataFrame] = None

            for ds_name in settings.datasets:
                settings.run_number = int(run)
                ds = build_dataset(
                    full_ds_name=ds_name,
                    batch_size=settings.batch,
                    model_input_shape=settings.model_input_shape,
                )

                loaded_ds_list.append(ds)

                (test_results_df,
                 ds_info_df) = handle_single_dataset(ds=ds, settings=settings)
                # combine single results with other results
                if single_model_test_df is None:
                    single_model_test_df = test_results_df
                else:
                    single_model_test_df = pd.concat(
                        [single_model_test_df, test_results_df])

                # combine single results with other results
                if ds_info_df_all is None:
                    ds_info_df_all = ds_info_df
                else:
                    ds_info_df_all = pd.concat([ds_info_df_all, ds_info_df])

            if settings.is_evaluating_model:
                save_bundled_model_evaluation(single_model_test_df, settings)

            if settings.is_compiling_evalulation:
                compile_model_evaluation(settings)
            # avoid OOM
            tf.keras.backend.clear_session()
            gc.collect()

            print("---")
            print(f"Done training/evaluating models from run {run}")
            print("---")

        # we cannot really recover from this state
        # so we have to end the program at this point
        print(
            "Done training and evaluating models loaded from parameter.json!")
        sys.exit(0)

    # normal case of user specified run number, run name, model and dataset
    elif settings.datasets is not None:
        single_model_test_df: Optional[pd.DataFrame] = None
        for ds_name in settings.datasets:
            ds = build_dataset(
                full_ds_name=ds_name,
                batch_size=settings.batch,
                model_input_shape=settings.model_input_shape,
            )

            loaded_ds_list.append(ds)

            (test_results_df,
             ds_info_df) = handle_single_dataset(ds=ds, settings=settings)
            # combine single results with other results
            if single_model_test_df is None:
                single_model_test_df = test_results_df
            else:
                single_model_test_df = pd.concat(
                    [single_model_test_df, test_results_df])

            # combine single results with other results
            if ds_info_df_all is None:
                ds_info_df_all = ds_info_df
            else:
                ds_info_df_all = pd.concat([ds_info_df_all, ds_info_df])

    if settings.is_compiling_attack_results:
        compile_attack_results(settings)

    if settings.is_generating_ds_info:
        save_bundled_ds_info_df(ds_info_df_all, settings)

    if settings.is_evaluating_model:
        save_bundled_model_evaluation(single_model_test_df, settings)

    if settings.is_compiling_evalulation:
        compile_model_evaluation(settings)

    if settings.model_name is not None and (settings.is_train_model or
                                            settings.is_running_amia_attack):
        mode: str = None
        if settings.is_train_model:
            mode = "model_train"
        else:
            mode = "model_attack"

        param_filepath = os.path.join(
            result_path,
            settings.model_name,
            settings.run_name,
            str(settings.run_number),
        )

        end_time = dt.now()
        elapsed = end_time - start_time
        elapsed_time_str = f"{elapsed.days}D - {elapsed.seconds // 3600}H - {elapsed.seconds // 60 % 60}M - {elapsed.seconds % 60}S"
        print(f"Execution took: {elapsed_time_str}")

        settings.execution_time = elapsed_time_str
        settings.save_settings_as_json(param_filepath,
                                       file_name_extension=mode)


def handle_single_dataset(
    ds: AbstractDataset, settings: RunSettings
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    result_df = None
    ds_info_df = None

    # generate ds_info before preprocessing dataset
    if settings.is_generating_ds_info:
        print("---------------------")
        print("Generating Dataset Info")
        print("---------------------")
        ds_info_df = generate_ds_info(
            ds=ds,
            force_ds_info_regen=settings.is_forcing_ds_info_regeneration,
        )

    # prepare dataset after generating ds_info!
    ds.prepare_datasets()

    if (settings.is_train_model or settings.is_evaluating_model
            or settings.is_running_amia_attack):
        model_save_path: str = os.path.join(
            model_path,
            settings.model_name,
            settings.run_name,
            str(settings.run_number),
            ds.dataset_name,
        )
        check_create_folder(model_save_path)
        model_save_file: str = os.path.join(model_save_path,
                                            f"{ds.dataset_name}.tf")
        model = load_model(model_path=model_save_file,
                           num_classes=ds.num_classes,
                           settings=settings)

        # set values for private training
        if type(model) is PrivateCNNModel:
            num_train_samples = int(len(ds.get_train_ds_as_numpy()[0]))
            model.set_privacy_parameter(
                epsilon=settings.privacy_epsilon,
                num_train_samples=num_train_samples,
                l2_norm_clip=settings.l2_norm_clip,
                num_microbatches=settings.num_microbatches,
                noise_multiplier=settings.noise_multiplier,
            )
            settings.noise_multiplier = model.noise_multiplier
            settings.delta = model.delta
            print(
                f"Setting private training parameter epsilon: {model.epsilon}, l2_norm_clip: {model.l2_norm_clip}, num_microbatches: {model.num_microbatches} and noise_multiplier: {model.noise_multiplier}."
            )

    if settings.is_train_model:
        print("---------------------")
        print("Training single model")
        print("---------------------")
        train_model(ds=ds, model=model, settings=settings)

    if settings.is_evaluating_model:
        print("---------------------")
        print("Loading and evaluate model")
        print("---------------------")
        result_df = load_and_test_model(ds, model)

    if settings.is_running_amia_attack:
        print("---------------------")
        print(
            "Running AMIA attack (train shadow models, calc statistics, run attack)"
        )
        print("---------------------")
        shadow_model_save_path: str = os.path.join(
            model_path,
            settings.model_name,
            settings.run_name,
            str(settings.run_number),
            "shadow-models",
            ds.dataset_name,
        )
        check_create_folder(shadow_model_save_path)

        run_amia_attack(
            ds=ds,
            model=model,
            settings=settings,
            shadow_model_save_path=shadow_model_save_path,
        )

    # avoid OOM
    tf.keras.backend.clear_session()
    gc.collect()

    return result_df, ds_info_df


def generate_ds_info(ds: AbstractDataset,
                     force_ds_info_regen: bool) -> pd.DataFrame:
    """Generate dataset info.

    Needs to be run before dataset preprocessing is called.

    Return:
    ------
    pd.Dataframe -> Dataframe to compare different datasets by single-value metrics.

    """
    ds.build_ds_info(force_regeneration=force_ds_info_regen)
    ds.save_ds_info_as_json()
    ds.create_data_histogram()
    return ds.get_ds_info_as_df()


def load_model(model_path: str, num_classes: int,
               settings: RunSettings) -> Model:
    model = None

    model_height = settings.model_input_shape[0]
    model_width = settings.model_input_shape[1]
    model_color_space = settings.model_input_shape[2]

    if settings.model_name == "cnn":
        model = CNNModel(
            model_name="cnn",
            img_height=model_height,
            img_width=model_width,
            color_channels=model_color_space,
            random_seed=settings.random_seed,
            num_classes=num_classes,
            batch_size=settings.batch,
            model_path=model_path,
            epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            adam_epsilon=settings.adam_epsilon,
            ema_momentum=settings.ema_momentum,
            weight_decay=settings.weight_decay,
        )
    elif settings.model_name == "private_cnn":
        model = PrivateCNNModel(
            model_name="private_cnn",
            img_height=model_height,
            img_width=model_width,
            color_channels=model_color_space,
            random_seed=settings.random_seed,
            num_classes=num_classes,
            batch_size=settings.batch,
            model_path=model_path,
            epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            adam_epsilon=settings.adam_epsilon,
            ema_momentum=settings.ema_momentum,
            weight_decay=settings.weight_decay,
        )
    return model


def train_model(ds: AbstractDataset, model: Model, settings: RunSettings):
    model.build_compile()
    model.print_summary()

    x, y = ds.convert_ds_to_one_hot_encoding(ds.ds_train, unbatch=True)
    test_x, test_y = ds.convert_ds_to_one_hot_encoding(ds.ds_test,
                                                       unbatch=True)

    model.train_model_from_numpy(x=x, y=y, val_x=test_x, val_y=test_y)

    model.save_model()

    train_history_folder = os.path.join(
        result_path,
        model.model_name,
        settings.run_name,
        str(settings.run_number),
        "single-model-train",
    )
    model.save_train_history(
        folder_name=train_history_folder,
        image_name=f"{ds.dataset_name}_model_train_history.png",
    )
    # avoid OOM
    tf.keras.backend.clear_session()
    gc.collect()


def load_and_test_model(ds: AbstractDataset, model: Model) -> pd.DataFrame:
    model.load_model()
    model.compile_model()
    model.print_summary()

    x, y = ds.convert_ds_to_one_hot_encoding(ds.ds_train, unbatch=True)
    test_x, test_y = ds.convert_ds_to_one_hot_encoding(ds.ds_test,
                                                       unbatch=True)

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
    df = df[[
        "name", "type", "accuracy", "f1-score", "precision", "recall", "loss"
    ]]
    return df


def run_amia_attack(
    ds: AbstractDataset,
    model: Model,
    settings: RunSettings,
    shadow_model_save_path: str,
):
    amia_result_path = os.path.join(result_path,
                                    settings.model_name, settings.run_name,
                                    str(settings.run_number))
    amia = AmiaAttack(
        model=model,
        ds=ds,
        num_shadow_models=settings.num_shadow_models,
        shadow_model_dir=shadow_model_save_path,
        result_path=amia_result_path,
        include_mia=settings.is_running_mia_attack,
    )
    amia.train_load_shadow_models(
        force_retraining=settings.is_force_model_retrain,
        force_recalculation=settings.is_force_stat_recalculation,
        num_microbatch=settings.num_microbatches,
    )
    amia.attack_shadow_models_amia()


def generate_privacy_report(settings: RunSettings, num_samples: int = 60000):
    print(
        f"Calculating privacy statement for given parameter (assuming {num_samples} train samples)..."
    )
    used_microbatching = True
    if settings.num_microbatches <= 1:
        used_microbatching = False

    settings.delta = compute_delta(num_samples)

    if settings.noise_multiplier is None:
        settings.noise_multiplier = compute_noise(
            num_train_samples=num_samples,
            batch_size=settings.batch,
            target_epsilon=settings.privacy_epsilon,
            epochs=settings.epochs,
            delta=settings.delta,
        )

    priv_report = compute_privacy(
        num_samples,
        settings.batch,
        settings.noise_multiplier,
        settings.epochs,
        settings.delta,
        used_microbatching=used_microbatching,
    )
    print(priv_report)
    sys.exit(0)


def run_args_parameter_check(settings: RunSettings):
    if settings.model_name is None:
        if not settings.is_generating_ds_info and (
                not settings.is_train_model
                or not settings.is_evaluating_model):
            print("No model was specified! Please provide a valid model name!")
            sys.exit(1)

    if (settings.is_train_model or settings.is_evaluating_model
            or settings.is_running_amia_attack):
        if settings.run_number is None and settings.run_name is None:
            print(
                "No run number and no run name specified! A run number is required when training/ attacking/ testing models! See help for more info."
            )
            sys.exit(1)

        if settings.model_name is None:
            print(
                "No model specified! A model is required when training/ attacking/ testing models!"
            )
            sys.exit(1)

        if settings.datasets is None and settings.run_name is None:
            print(
                "No datasets specified and no run name specified! A datasets is required when training/ attacking/ testing models! See help for more info."
            )
            sys.exit(1)

    if settings.is_generating_ds_info and settings.datasets is None:
        print(
            "No datasets specified! A datasets is required when training/ attacking/ testing models!"
        )
        sys.exit(1)


def compile_attack_results(settings: RunSettings):
    print("---------------------")
    print("Compiling attack results")
    print("---------------------")
    analyser = AttackAnalyser(
        settings=settings,
        result_path=result_path,
        model_path=model_path,
    )
    analyser.compile_attack_results(AttackType.LIRA)

    if settings.is_running_mia_attack:
        analyser.compile_attack_results(AttackType.MIA)


def save_bundled_ds_info_df(ds_info_df: pd.DataFrame, settings: RunSettings):
    print("---------------------")
    print("Saving Bundled Dataset Info")
    print("---------------------")
    print(ds_info_df)
    ds_info_df_file = os.path.join(
        ds_info_path, f'dataframe_{"-".join(settings.datasets)}_ds_info.csv')
    save_dataframe(ds_info_df, ds_info_df_file)


def save_bundled_model_evaluation(model_test_df: pd.DataFrame,
                                  settings: RunSettings):
    # save result_df with model's accuracy, loss, etc. gathered as csv
    result_df_filename = os.path.join(
        result_path,
        settings.model_name,
        settings.run_name,
        str(settings.run_number),
        "single-model-train",
        f'{"-".join(settings.datasets)}_model_predict_results.csv',
    )
    check_create_folder(os.path.dirname(result_df_filename))

    save_dataframe(df=model_test_df,
                   filename=result_df_filename,
                   use_index=False)


def compile_model_evaluation(settings: RunSettings):
    print("---------------------")
    print("Compiling model evaluation")
    print("---------------------")
    cwd = os.getcwd()
    res_path = os.path.join(cwd, result_path)
    analyser = UtilityAnalyser(result_path=res_path, settings=settings)

    analyser.analyse_utility()


if __name__ == "__main__":
    main()
