import matplotlib.pyplot as plt

import tensorflow as tf


def visualize_training(history: tf.keras.callbacks.History, epochs: int,
                       img_name: str = "results.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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

    return
