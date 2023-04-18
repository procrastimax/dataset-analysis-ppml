from ppml_datasets import MnistDataset

from model import CNNModel


def main():
    cnn_model = CNNModel(img_height=24, img_width=24, color_channels=3, num_classes=10)
    cnn_model.build_compile()
    cnn_model.print_summary()


if __name__ == "__main__":
    main()
