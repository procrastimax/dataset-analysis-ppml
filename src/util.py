import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Any
import os
import pickle
import pandas as pd
from ppml_datasets.utils import check_create_folder


def visualize_training(history: tf.keras.callbacks.History,
                       img_name: str = "results.png"):
    print(f"Saving trainings results to: {img_name}")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
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


def save_dataframe(df: pd.DataFrame, filename: str, sep: str = ",", use_index: bool = True, header: bool = True):
    print(f"Saving dataframe as csv: {filename}")
    df.to_csv(path_or_buf=filename, header=header, index=use_index, sep=sep)


def plot_curve_with_area(x, y, xlabel, ylabel, ax, label, title: str, use_log_scale: bool):
    ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
    ax.plot(x, y, lw=2, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if use_log_scale:
        ax.set(aspect=1, xscale='log', yscale='log')
    ax.title.set_text(title)


def plot_histogram(counts: np.array, bins: np.array, filename: str, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(5, 5))
    plt.stairs(counts, bins, fill=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)
    plt.close()
