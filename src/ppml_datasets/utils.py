import json
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def check_create_folder(dir: str):
    """Check if a folder exists on the current file, if not, this function creates that folder."""
    base_dir = os.path.realpath(os.getcwd())
    check_dir = os.path.join(base_dir, dir)
    if not os.path.exists(check_dir):
        print(f"Directory {check_dir} does not exist, creating it")
        os.makedirs(check_dir)


def get_img(x, y):
    """Load image from path and return together with label."""
    path = x
    label = y
    # load the raw data from the file as a string
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return img, label


def visualize_training(
    history: tf.keras.callbacks.History, img_name: str = "results.png"
):
    print(f"Saving training figure {img_name}")
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
    plt.close()


def visualize_data(ds: tf.data.Dataset, file_name: str = "data_vis.png"):
    class_names = None

    if hasattr(ds, "class_names"):
        print("dataset has set class names")
        print(ds.class_names)
        class_names = ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            if class_names is not None:
                plt.title(class_names[labels[i]])
            else:
                plt.title(i)

    plt.axis("off")
    plt.savefig(file_name)
    plt.close()


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
    plt.close()


def filter_labels(y, allowed_classes: list):
    """Return `True` if `y` belongs to `allowed_classes` list else `False`.

    Example usage:
        dataset.filter(lambda s: filter_classes(s['label'], [0,1,2])) # as dict
        dataset.filter(lambda x, y: filter_classes(y, [0,1,2])) # as_supervised
    """
    allowed_classes = tf.constant(allowed_classes)
    isallowed = tf.equal(allowed_classes, tf.cast(y, allowed_classes.dtype))
    reduced_sum = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced_sum, tf.constant(0.0))


def get_ds_as_numpy(
    ds: tf.data.Dataset, unbatch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if ds is None:
        print("Cannot convert dataset to numpy arrays! Dataset is not initialized!")
        return

    values = []
    labels = []

    if unbatch:
        for x, y in ds.unbatch().as_numpy_iterator():
            values.append(x)
            labels.append(y)
    else:
        for x, y in ds.as_numpy_iterator():
            values.append(x)
            labels.append(y)
    return (np.asarray(values), np.asarray(labels))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_dict_as_json(dict_object: Dict[Any, Any], filename: str):
    print(f"Saving dict as json file {filename}")
    with open(filename, "w") as f:
        json.dump(dict_object, f, cls=NpEncoder, indent=2, ensure_ascii=True)


def load_dict_from_json(filename: str) -> Optional[Dict[Any, Any]]:
    print(f"Loading dict from json file {filename}")
    if not os.path.isfile(filename):
        print(f"File {filename} does not exist! Returning None.")
        return None

    with open(filename, "r") as f:
        return json.load(f)
