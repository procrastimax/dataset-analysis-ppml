import sys
from typing import Any, Dict, List, Tuple

from ppml_datasets.abstract_dataset_handler import AbstractDataset
from ppml_datasets.datasets.cifar10 import build_cifar10
from ppml_datasets.datasets.cifar100 import build_cifar100
from ppml_datasets.datasets.emnist import (EMNISTDigitsManyBalancedDataset,
                                           EMNISTDigitsNormalBalancedDataset,
                                           EMNISTLargeUnbalancedDataset,
                                           EMNISTLettersBalancedDataset,
                                           EMNISTMediumBalancedDataset,
                                           EMNISTMediumUnbalancedDataset)
from ppml_datasets.datasets.fmnist import build_fmnist
from ppml_datasets.datasets.mnist import build_mnist
from ppml_datasets.datasets.svhn import build_svhn


def parse_dataset_name_parameter(ds_mods: List[str]) -> Dict[str, List[Any]]:
    # check if there are really modifications
    if len(ds_mods) == 0:
        return {}

    mod_dict = {}
    for mod in ds_mods:
        if mod.startswith("c"):
            class_size = int(mod.removeprefix("c"))
            mod_dict["c"] = [class_size]

        if mod.startswith("gray"):
            mod_dict["gray"] = [True]

        if mod.startswith("i"):
            mod = mod.removeprefix("i")
            imbalance_ratio = float(mod[1:])
            imbalance_mode = mod[0]

            if imbalance_mode not in ("L", "N"):
                print(f"Unknown imbalance mode: {imbalance_mode}")
                continue
            mod_dict["i"] = [imbalance_mode, imbalance_ratio]

        if mod.startswith("n"):
            num_new_classes = int(mod.removeprefix("n"))
            mod_dict["n"] = [num_new_classes]

    return mod_dict


def build_dataset(
    full_ds_name: str, batch_size: int, model_input_shape: Tuple[int, int, int]
) -> AbstractDataset:
    ds = None
    parameterized_name = full_ds_name.split("_")

    # exluce dataset name from parameter parsing
    mod_params = parse_dataset_name_parameter(parameterized_name[1:])

    if parameterized_name[0] == "mnist":
        ds = build_mnist(model_input_shape, batch_size, mod_params)

    elif parameterized_name[0] == "fmnist":
        ds = build_fmnist(model_input_shape, batch_size, mod_params)

    elif parameterized_name[0] == "cifar10":
        ds = build_cifar10(model_input_shape, batch_size, mod_params)

    elif parameterized_name[0] == "svhn":
        ds = build_svhn(model_input_shape, batch_size, mod_params)

    elif parameterized_name[0] == "cifar100":
        ds = build_cifar100(model_input_shape, batch_size, mod_params)

    # the EMNIST datasets and their variations
    elif parameterized_name[0] == "emnist-large-unbalanced":
        ds = EMNISTLargeUnbalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()
    elif parameterized_name[0] == "emnist-medium-unbalanced":
        ds = EMNISTMediumUnbalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()
    elif parameterized_name[0] == "emnist-medium-balanced":
        ds = EMNISTMediumBalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()
    elif parameterized_name[0] == "emnist-letters-balanced":
        ds = EMNISTLettersBalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()
    elif parameterized_name[0] == "emnist-digits-balanced":
        ds = EMNISTDigitsManyBalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()
    elif parameterized_name[0] == "emnist-mnist-balanced":
        ds = EMNISTDigitsNormalBalancedDataset(
            model_img_shape=model_input_shape,
            builds_ds_info=False,
            batch_size=batch_size,
            augment_train=False,
        )
        ds.load_dataset()

    else:
        print(
            f"The requested: {full_ds_name} dataset does not exist or is not implemented!"
        )
        return None

    if ds is None:
        print("Could not create valid dataset! Exiting...")
        sys.exit(1)

    return ds
