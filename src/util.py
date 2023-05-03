import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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
