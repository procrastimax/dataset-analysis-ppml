# Dataset Anaylsis for Privacy Preserving Machine Learning

# Usage
```
Dataset Analysis for Privacy-Preserving-Machine-Learning [-h] -d {mnist,fmnist,cifar10,cifar10gray} [{mnist,fmnist,cifar10,cifar10gray} ...] -r R [-s N] [--train-single-model] [--load-test-single-model] [--run-amia-attack] [--generate-results]
                                                                [--force-model-retrain] [--force-stat-recalculation] [--generate-ds-info]

A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.

optional arguments:
  -h, --help            show this help message and exit
  -d {mnist,fmnist,cifar10,cifar10gray} [{mnist,fmnist,cifar10,cifar10gray} ...], --datasets {mnist,fmnist,cifar10,cifar10gray} [{mnist,fmnist,cifar10,cifar10gray} ...]
                        Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here.
  -r R, --run-number R  The run number to be used for training models, loading or saving results.
  -s N, --shadow-model-number N
                        The number of shadow models to be trained if '--train-shadow-models' is set.
  --train-single-model  If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number.
  --load-test-single-model
                        If this flag is set, a single model is loaded based on run number and dataset name. Then predictions are run on the test and train dataset.
  --run-amia-attack     If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved. This can be seen as Step 2 in the analysis pipeline.
  --generate-results    If this flag is set, all saved results are compiled and compared with each other, allowing dataset comparison. This can be seen as Step 3 in the analysis pipeline.
  --force-model-retrain
                        If this flag is set, the shadow models, even if they already exist.
  --force-stat-recalculation
                        If this flag is set, the statistics are recalucated on the shadow models.
  --generate-ds-info    If this flag is set, dataset infos are generated and saved.
```
