from ppml_datasets import MnistDataset


def main():
    ds = MnistDataset([1, 1, 1], builds_ds_info=False)
    ds.load_dataset()
    ds.build_ds_info()
    ds.prepare_datasets()
    print("done loading!")
    print(ds.ds_info)


if __name__ == "__main__":
    main()
