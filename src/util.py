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

    class_names = None

    if hasattr(ds, "class_names"):
        print("dataset has set class names")
        print(ds.class_names)
        class_names = ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            print(images[i].numpy)
            plt.imshow(images[i].numpy().astype("uint8"))
            if class_names is not None:
                plt.title(class_names[labels[i]])
            else:
                plt.title(i)
            plt.axis("off")
            plt.savefig(file_name)
