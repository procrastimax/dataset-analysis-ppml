# Dataset Anaylsis for Privacy Preserving Machine Learning

This project provides a framework to evaluate the influence of different image dataset characteristics on CNN-based machine learning classification in regard to model utility and model privacy.
This framework enables to train multiple models trained on pre-selected MNIST-like datasets with different budgets.
The private models (epsilon != infty) utilize DP-Adam to establish differential privacy.
All models can be attacked with the classic MIA Threshold attack or with the state-of-the-art LiRA attack.
The model training results and the attack results allow a comparison of different dataset characteristics and their influence on the model behavior.

There are two types of dataset characteristics that can be examined with this framework are data-level and dataset-level characteristics.
Data-level characteristics are characteristics that directly correlate to the data used in the dataset, i.e., entropy, class separability, compression ratios, etc.
While the dataset-level characteristics concern the dataset structure like number of classes, dataset size and imbalance.
The dataset-level characteristics can be actively modified to specifically compare the same origin dataset with modified versions of it, where some of these characteristics change.

## Usage
```
Dataset Analysis for Privacy-Preserving-Machine-Learning [-h] [-d D [D ...]] [-m M] [-r R] [-n N] [-s S] [--epochs EPOCHS] [-l L] [-e E] [-ae ADAM_EPSILON] [--batch-size BATCH_SIZE] [-wd WEIGHT_DECAY] [-ema MOMENTUM] [-c C] [-b B] [-np NOISE_MULTIPLIER] [-tm]
                                                                [-em] [-ce] [--run-amia-attack] [-ca] [-ar AR] [--force-model-retrain] [--force-stat-recalculation] [--generate-ds-info] [--force-ds-info-regeneration] [--include-mia] [-p] [--xName XNAME]
                                                                [--xValues XVALUES [XVALUES ...]]

A toolbox to analyse the influence of dataset characteristics on the performance of algorithm pertubation in PPML.

optional arguments:
  -h, --help            show this help message and exit
  -d D [D ...], --datasets D [D ...]
                        Which datasets to load before running the other steps. Multiple datasets can be specified, but at least one needs to be passed here. Available datasets are: mnist, fmnist, cifar10, cifar100, svhn,
                        emnist-(large|medium|letters|digits|mnist)-(unbalanced|balanced). With modifications _cX (class size), _i[L/N]Y (imbalance), _nX (number of classes), _gray.
  -m M, --model M       Specify which model should be used for training/ attacking. Only one can be selected!
  -r R, --run-number R  The run number to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results.
  -n N, --run-name N    The run name to be used for training models, loading or saving results. This flag is theoretically not needed if you only want to generate ds-info results. The naming hierarchy here is: model_name/run_name/run_number.
  -s S, --shadow-models S
                        The number of shadow models to be trained.
  --epochs EPOCHS       The number of epochs the model should be trained on.
  -l L, --learning-rate L
                        The learning rate used for training models.
  -e E, --epsilon E     The desired epsilon value for DP-SGD learning. Can be any value: 0.1, 1, 10, ...
  -ae ADAM_EPSILON, --adam-epsilon ADAM_EPSILON
                        The epsilon hat value for the Adam optimizer.
  --batch-size BATCH_SIZE
                        Size of batch used for training.
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        The weight decay used in the Adam optimizer.
  -ema MOMENTUM, --momentum MOMENTUM
                        Momentum value used for Adam's EMA when training the models. If set, EMA in Adam is activated.
  -c C, --l2-norm-clip C
                        The L2 norm clip value set for private training models.
  -b B, --microbatches B
                        Number of microbatches used for private training.
  -np NOISE_MULTIPLIER, --noise-multiplier NOISE_MULTIPLIER
                        A fix set noise multiplier for DP-SGD.
  -tm, --train-model    If this flag is set, a single model is trained on the given datasets (respecting train_ds, val_ds & test_ds). This always overrides a previously trained model on the same dataset name and run number. If no dataset name and no run number but the
                        run name is given, it is assumed that a single model from a series of attacks shall get trained by parsing the datasets from 'parameter.json' files in each runs. In this scenario, all runs are included.
  -em, --evaluate-model
                        If this flag is set, a single model is loaded based on run number, run name, model name and dataset name. Then predictions are run on the test and train dataset to evaluate the model. If no dataset name and no run number but the run name is given,
                        models trained from the 'parameter.json' file are evaluated.
  -ce, --compile-evaluation
                        If this flag is set, the program compiles all single model evaluations from different run numbers to a single file.
  --run-amia-attack     If this flag is set, an Advanced MIA attack is run on the trained shadow models and the results are saved.
  -ca, --compile-attack-results
                        If this flag is set, all saved attack results are compiled and compared with each other, allowing dataset comparison.
  -ar AR, --analysis-run-numbers AR
                        The run numbers (1,2,3,4) or range of run numbers (1-4) or a combination of both (1,2,3-5) to be used for result compilation.
  --force-model-retrain
                        If this flag is set, the shadow models, even if they already exist.
  --force-stat-recalculation
                        If this flag is set, the statistics are recalucated on the shadow models.
  --generate-ds-info    If this flag is set, dataset infos are generated and saved.
  --force-ds-info-regeneration
                        If this flag is set, the whole ds-info dict is not loaded from a json file but regenerated from scratch.
  --include-mia         If this flag is set, then the mia attack is also used during attacking and mia related results/ graphics are produced during result generation.
  -p, --generate-privacy-report
                        Dont train/load anything, just generate a privacy report for the given values.
  --xName XNAME         Name of the X axis for graph generation
  --xValues XVALUES [XVALUES ...]
                        X axis values for graph generation
```

## Build Docker Container
It is advised to run the framework within a docker container to create a reproducible experiment environment.
The following steps describe how to build and run your own container.

- setup proper Docker environment with nvidia-docker-container support
- build docker image from Dockerfile: `sudo docker build -t dataset-analysis .`
- inspect container and pass GPU: `sudo docker run -it --rm --gpus all dataset-analysis bash -c 'nvidia-smi'`
- run code in container: `sudo docker run --rm --gpus all dataset-analysis python3.9 src/main.py --help`

## Dataset-Level Characteristics
The dataset-level characteristics are directly parsed from the dataset name that is given to the application as CLI parameters.
For example: `python src/main.py -d mnist -tm -em -m private_cnn -e 30`, trains and evaulates a single private CNN model with a privacy budget of 30. The dataset used to train the dataset is the MNIST dataset.
With a slight dataset name modification, different dataset-level characteristics can be applied.
For example: `python src/main.py -d mnist_n5 -tm -em -m cnn --run-amia-attack`, trains and evaluates a single non-private CNN and runs the LiRA attack on the model. The dataset used for training and attacking the model is a modified version of the MNIST dataset containing only 5 classes (instead of 10).
These dataset-level modifications can be chained.

A full list of modifications is provided below:
- mnist_nX -> set number of dataset classes to X
- mnist_cX -> set number of samples per class to X. Creates a perfectly balanced dataset since all classes are reduced to the same number of samples.
- mnist_iLX -> creates an imbalanced dataset version with an imbalance factor of X in linear mode. A higher imbalance mode indicates a higher dataset imbalance.
- mnist_iNX -> creates an imbalanced dataset version with an imbalance factor of X in normal mode. A higher imbalance mode indicates a higher dataset imbalance. (normal and linear mode differ in how they apply the dataset imbalance)
- cifar10_gray -> creates a grayscaled version of a dataset containing RGB color images.
