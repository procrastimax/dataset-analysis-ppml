from ppml_datasets import MnistDataset

from model import CNNModel
from util import visualize_data


def main():
    # cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    # cnn_model.build_compile()
    # cnn_model.print_summary()

    mnist = MnistDataset([28, 28, 3], builds_ds_info=False)
    mnist.load_dataset()
    mnist.ds_train = mnist.prepare_ds(mnist.ds_train, resize_rescale=True,
                                      img_shape=[28, 28, 3], convert_rgb=True,
                                      batch_size=32,
                                      preprocessing_func=None,
                                      shuffle=False, augment=False)
    visualize_data(mnist.ds_train)


if __name__ == "__main__":
    main()
