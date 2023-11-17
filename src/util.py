import os
import pickle
from typing import Any, List

import dp_accounting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy_statement,
)
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise as tfp_compute_noise,
)

from ppml_datasets.utils import check_create_folder


def visualize_training(
    history: tf.keras.callbacks.History, img_name: str = "results.png"
):
    print(f"Saving trainings results to: {img_name}")
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(img_name)


def visualize_data(ds: tf.data.Dataset, file_name: str = "data_vis.png"):
    plt.figure(figsize=(10, 10))

    ds_it = ds.as_numpy_iterator()

    for i in range(9):
        (image, label) = ds_it.next()
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[0])
        plt.title(label[0])
    plt.axis("off")
    plt.savefig(file_name)


def visualize_data_np(x: np.ndarray, y: np.ndarray, file_name: str = "data_vis_np.png"):
    plt.figure(figsize=(10, 10))

    counter: int = 0

    for images, labels in zip(x, y):
        plt.subplot(3, 3, counter + 1)
        plt.imshow(images)
        plt.title(labels)

        counter += 1
        if counter >= 9:
            break
    plt.axis("off")
    plt.savefig(file_name)


def pickle_object(full_filename: str, object: Any, overwrite_if_exist: bool = True):
    dirname = os.path.dirname(full_filename)
    check_create_folder(dirname)

    if not overwrite_if_exist:
        # check before writing so we dont overwrite stuff
        if os.path.isfile(full_filename):
            print(f"File {full_filename} already exist, not overwriting it!")
            return

    with open(full_filename, "wb") as f:
        pickle.dump(object, f)
        print(f"Saved .pckl file {full_filename}")


def unpickle_object(full_filename: str) -> Any:
    if os.path.isfile(full_filename):
        with open(full_filename, "rb") as f:
            object = pickle.load(f)
            print(f"Loaded .pckl file: {full_filename}")
            return object
    else:
        print(f"Cannot load object from pickel! {full_filename} does not exist!")
        return None


def find_nearest(array, value) -> (int, float):
    """Find nearest value in array given another value.

    Return:
    ------
    (int, float) -> the found index and its value in the array

    """
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    sep: str = ",",
    use_index: bool = True,
    header: bool = True,
):
    print(f"Saving dataframe as csv: {filename}")
    df.to_csv(
        path_or_buf=filename,
        header=header,
        index=use_index,
        sep=sep,
        float_format="%.4f",
    )


def plot_curve_with_area(
    x, y, xlabel, ylabel, ax, label, title: str, use_log_scale: bool
):
    ax.plot([0, 1], [0, 1], "k-", lw=1.0)
    ax.plot(x, y, lw=2, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if use_log_scale:
        ax.set(aspect=1, xscale="log", yscale="log")
    ax.title.set_text(title)


def plot_histogram(
    counts: np.array,
    bins: np.array,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt.figure(figsize=(5, 3), layout="constrained")
    plt.stairs(counts, bins, fill=True)
    # plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)
    plt.close()


def compute_privacy(
    n: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float,
    used_microbatching: bool,
) -> str:
    """Calculate value of epsilon for given DP-SGD parameters."""
    return compute_dp_sgd_privacy_statement(
        number_of_examples=n,
        batch_size=batch_size,
        num_epochs=epochs,
        noise_multiplier=noise_multiplier,
        delta=delta,
        used_microbatching=used_microbatching,
    )


def compute_numerical_epsilon(
    steps: int, noise_multiplier: float, batch_size: int, num_samples: int
) -> float:
    """Computes epsilon value for given hyperparameters.

    Code copied from: https://github.com/tensorflow/privacy/blob/v0.8.10/tutorials/mnist_dpsgd_tutorial_keras_model.py
    """
    if noise_multiplier == 0.0:
        return float("inf")

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = batch_size / num_samples

    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )

    accountant.compose(event)

    delta = compute_delta(num_samples)
    return accountant.get_epsilon(target_delta=delta)


def compute_noise(
    num_train_samples: int,
    batch_size: int,
    target_epsilon: float,
    epochs: int,
    delta: float,
    min_noise: float = 1e-5,
) -> float:
    """Calculate noise for given training hyperparameters."""
    return tfp_compute_noise(
        num_train_samples, batch_size, target_epsilon, epochs, delta, min_noise
    )


def compute_delta(num_train_samples: int):
    """Calculate Delta for given training dataset size n.

    Code from Lucas Lange: https://github.com/luckyos-code/mia-covid/blob/main/mia_covid/evaluation.py
    """
    # delta should be one magnitude lower than inverse of training set size: 1/n
    # e.g. 1e-5 for n=60.000
    # take 1e-x, were x is the magnitude of training set size
    delta = np.power(
        10, -float(len(str(num_train_samples)))
    )  # remove all trailing decimals
    return delta


def get_run_numbers(run_result_folder: str) -> List[int]:
    """Scan run result folder for available run numbers."""
    run_numbers: List[int] = []
    folders = os.scandir(run_result_folder)
    for entry in folders:
        if entry.is_dir():
            if entry.name.isnumeric():
                run_numbers.append(int(entry.name))

    run_numbers.sort()
    return run_numbers
