# Dataset Anaylsis for Privacy Preserving Machine Learning

# Usage
```
Dataset Analysis for Privacy-Preserving-Machine-Learning [-h]
                                                                [-d {mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} [{mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} ...]]
                                                                [-m {small_cnn,private_small_cnn}] [-r R] [-s N] [--train-single-model] [--epochs EPOCHS]
                                                                [-l LEARNING_RATE] [--momentum MOMENTUM] [-c L2_NORM_CLIP] [-b MICROBATCHES]
                                                                [--batch-size BATCH_SIZE] [--load-test-single-model] [--run-amia-attack] [--generate-results]
                                                                [--force-model-retrain] [--force-stat-recalculation] [--generate-ds-info]
                                                                [--force-ds-info-regeneration] [--include-mia] [-e EPSILON]

A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.

optional arguments:
  -h, --help            show this help message and exit
  -d {mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} [{mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} ...], --datasets {mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} [{mnist,mnist_c5000,fmnist,fmnist_c5000,cifar10,cifar10gray} ...]
                        Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here.
  -m {small_cnn,private_small_cnn}, --model {small_cnn,private_small_cnn}
                        Specify which model should be used for training/ attacking. Only one can be selected!
  -r R, --run-number R  The run number to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to
                        generate ds-info results.
  -s N, --shadow-model-number N
                        The number of shadow models to be trained if '--train-shadow-models' is set.
  --train-single-model  If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a
                        previously trained model on the same dataset name and run number.
  --epochs EPOCHS       The number of epochs the model should be trained on.
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        The learning rate used for training models.
  --momentum MOMENTUM   Momentum value used for training the models.
  -c L2_NORM_CLIP, --l2-norm-clip L2_NORM_CLIP
                        The L2 norm clip value set for private training models.
  -b MICROBATCHES, --microbatches MICROBATCHES
                        Number of microbatches used for private training.
  --batch-size BATCH_SIZE
                        Size of batch used for training.
  --load-test-single-model
                        If this flag is set, a single model is loaded based on run number and dataset name. Then predictions are run on the test and train
                        dataset.
  --run-amia-attack     If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved.
  --generate-results    If this flag is set, all saved results are compiled and compared with each other, allowing dataset comparison.
  --force-model-retrain
                        If this flag is set, the shadow models, even if they already exist.
  --force-stat-recalculation
                        If this flag is set, the statistics are recalucated on the shadow models.
  --generate-ds-info    If this flag is set, dataset infos are generated and saved.
  --force-ds-info-regeneration
                        If this flag is set, the whole ds-info dict is not loaded from a json file but regenerated from scratch.
  --include-mia         If this flag is set, then the mia attack is also used during attacking and mia related results/ graphics are produced during result
                        generation.
  -e EPSILON, --epsilon EPSILON
                        The desired epsilon value for DP-SGD learning. Can be any value: 0, 0.1, 1, 10, None (if not set)
```

# Build Dev-Docker Container
- setup proper Docker environment with nvidia-docker-container support
- build docker image from Dockerfile: `sudo docker build -t dataset-analysis .`
- inspect container and pass GPU: `sudo docker run -it --rm --gpus all dataset-analysis bash -c 'nvidia-smi'`
- run code in container: `sudo docker run --rm --gpus all dataset-analysis python3.9 src/main.py --help`
