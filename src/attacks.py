import tensorflow as tf
from scipy import special
from typing import Optional
from model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
import numpy as np

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, SlicingSpec, AttackType, SingleSliceSpec, SlicingFeature
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.dataset_slicing import get_slice
from tensorflow_privacy.privacy.privacy_tests import utils
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia

import matplotlib.pyplot as plt

import functools
from os import sys
import os
import gc
import copy


class MiaAttack():
    """Implementation for multi class mia attack.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used
    """

    def __init__(self, model: CNNModel, dataset: AbstractDataset, num_classes: Optional[int] = None):
        """Initialize MiaAttack class.

        Paramter:
        ---------
        model : CNNModel
        dataset : AbstractDataset
        num_classes : int | None - needed for visualizing the membership Probability
        """
        self.cnn_model: CNNModel = model

        if dataset.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            sys.exit(1)

        if dataset.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            sys.exit(1)

        self.ds_classes: int = 0

        self.ds = dataset
        if self.ds.ds_info is not None and "classes" in self.ds.ds_info.keys():
            self.ds_classes: int = self.ds.ds_info["classes"]
        else:
            if num_classes is not None:
                self.ds_classes = num_classes
            else:
                print("ERROR: Number of classes was not specified by either the dataset nor the the classe's initialization")
                sys.exit(1)

        self.logits_train = None
        self.logits_test = None

        self.prob_train = None
        self.prob_test = None

        self.scce = tf.keras.backend.sparse_categorical_crossentropy
        self.constant = tf.keras.backend.constant

        self.loss_train = None
        self.loss_test = None

        self.train_labels = None
        self.test_labels = None

        self.train_images = None
        self.test_images = None

        self.input: AttackInputData = AttackInputData()

    def initialize_data(self):
        """Initialize and calculate logits, probabilities and loss values for training and test sets."""
        print("Predict on train data...")
        self.logits_train = self.cnn_model.model.predict(self.ds.ds_attack_train, batch_size=self.cnn_model.batch_size)

        print("Predict on unseen test data...")
        self.logits_test = self.cnn_model.model.predict(self.ds.ds_attack_test, batch_size=self.cnn_model.batch_size)

        print("Apply softmax to get probabilities from logits")
        self.prob_train = special.softmax(self.logits_train, axis=1)
        self.prob_test = special.softmax(self.logits_test, axis=1)

        print("Get labels from dataset")
        self.train_labels = self.ds.get_attack_train_labels()
        self.test_labels = self.ds.get_attack_test_labels()

        print("Get images from dataset")
        self.train_images = self.ds.get_attack_train_values()
        self.test_images = self.ds.get_attack_test_values()

        print("Compute losses")
        self.loss_train = self.scce(self.constant(self.train_labels), self.constant(self.prob_train), from_logits=False).numpy()
        self.loss_test = self.scce(self.constant(self.test_labels), self.constant(self.prob_test), from_logits=False).numpy()

        # Suppose we have the labels as integers starting from 0
        # labels_train  shape: (n_train, )
        # labels_test  shape: (n_test, )

        # Evaluate your model on training and test examples to get
        # logits_train  shape: (n_train, n_classes)
        # logits_test  shape: (n_test, n_classes)
        # loss_train  shape: (n_train, )
        # loss_test  shape: (n_test, )

        self.input = AttackInputData(
            logits_train=self.logits_train,
            logits_test=self.logits_test,
            loss_train=self.loss_train,
            loss_test=self.loss_test,
            # probs_train=self.prob_train,
            # probs_test=self.prob_test,
            labels_train=self.train_labels,
            labels_test=self.test_labels
        )

    def run_mia_attack(self):
        print("Running MIA attacks")

        if self.input is None:
            print("Error: Please run 'initialize_data()' before! Uninitialized AttackInputData!")
            sys.exit(1)

        attack_types = [AttackType.THRESHOLD_ATTACK,
                        AttackType.LOGISTIC_REGRESSION,
                        AttackType.RANDOM_FOREST,
                        AttackType.THRESHOLD_ENTROPY_ATTACK,
                        AttackType.K_NEAREST_NEIGHBORS,
                        AttackType.MULTI_LAYERED_PERCEPTRON]

        slicing_spec = SlicingSpec(entire_dataset=True,
                                   by_class=True,
                                   by_percentiles=False,
                                   by_classification_correctness=True)

        # run attacks for different data slices
        attacks_result = mia.run_attacks(self.input,
                                         attack_types=attack_types,
                                         slicing_spec=slicing_spec)

        # Print a user-friendly summary of the attacks
        print(attacks_result.summary(by_slices=True))

        # Plot the ROC curve of the best classifier
        fig = plotting.plot_roc_curve(
            attacks_result.get_result_with_max_auc().roc_curve)
        fig.savefig("mia_attcks.png")


class MembershipProbability():
    """Implementation for calculating membership probability.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used
    """

    def __init__(self, model: CNNModel, dataset: AbstractDataset, num_classes: Optional[int] = None):
        """Initialize MiaAttack class.

        Paramter:
        ---------
        model : CNNModel
        dataset : AbstractDataset
        num_classes : int | None - needed for visualizing the membership Probability
        """
        self.cnn_model: CNNModel = model

        if dataset.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            sys.exit(1)

        if dataset.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            sys.exit(1)

        self.ds_classes: int = 0

        self.ds = dataset
        if self.ds.ds_info is not None and "classes" in self.ds.ds_info.keys():
            self.ds_classes: int = self.ds.ds_info["classes"]
        else:
            if num_classes is not None:
                self.ds_classes = num_classes
            else:
                print("ERROR: Number of classes was not specified by either the dataset nor the the classe's initialization")
                sys.exit(1)

        self.logits_train = None
        self.logits_test = None

        self.prob_train = None
        self.prob_test = None

        self.scce = tf.keras.backend.sparse_categorical_crossentropy
        self.constant = tf.keras.backend.constant

        self.loss_train = None
        self.loss_test = None

        self.train_labels = None
        self.test_labels = None

        self.train_images = None
        self.test_images = None

        self.input: AttackInputData = AttackInputData()

    def initialize_data(self):
        """Initialize and calculate logits, probabilities and loss values for training and test sets."""
        print("Predict on train data...")
        self.logits_train = self.cnn_model.model.predict(self.ds.ds_attack_train, batch_size=self.cnn_model.batch_size)

        print("Predict on unseen test data...")
        self.logits_test = self.cnn_model.model.predict(self.ds.ds_attack_test, batch_size=self.cnn_model.batch_size)

        print("Apply softmax to get probabilities from logits")
        self.prob_train = special.softmax(self.logits_train, axis=1)
        self.prob_test = special.softmax(self.logits_test, axis=1)

        print("Get labels from dataset")
        self.train_labels = self.ds.get_attack_train_labels()
        self.test_labels = self.ds.get_attack_test_labels()

        print("Get images from dataset")
        self.train_images = self.ds.get_attack_train_values()
        self.test_images = self.ds.get_attack_test_values()

        print("Compute losses")
        self.loss_train = self.scce(self.constant(self.train_labels), self.constant(self.prob_train), from_logits=False).numpy()
        self.loss_test = self.scce(self.constant(self.test_labels), self.constant(self.prob_test), from_logits=False).numpy()

        self.input = AttackInputData(
            logits_train=self.logits_train,
            logits_test=self.logits_test,
            loss_train=self.loss_train,
            loss_test=self.loss_test,
            labels_train=self.train_labels,
            labels_test=self.test_labels
        )

    def calc_membership_probability(self, plot_training_samples: bool = True, num_images: int = 5):
        """Calculate Membership Probability also called Privacy Risk Score."""
        print("Calculating membership probability")

        if self.input is None:
            print("Error: Please run 'initialize_data()' before! Uninitialized AttackInputData!")
            sys.exit(1)

        slicing_spec = SlicingSpec(entire_dataset=True,
                                   by_class=True,
                                   by_percentiles=False,
                                   by_classification_correctness=False)  # setting this to True, somehow does not work

        membership_probability_results = mia.run_membership_probability_analysis(self.input, slicing_spec=slicing_spec)
        print(membership_probability_results.summary(threshold_list=[1, 0.9, 0.8, 0.7, 0.6, 0.5]))

        if plot_training_samples:
            print("Generating images to show high risk/ low risk training images")
            self._plot_training_samples(num_images)

    def _plot_training_samples(self, num_images: int = 5):
        for c in range(self.ds_classes):
            print(f"For data class: {c}")
            class_slice_spec = SingleSliceSpec(SlicingFeature.CLASS, c)
            class_input_slice = get_slice(self.input, class_slice_spec)
            class_dataset_idx = np.argwhere(self.input.labels_train == c).flatten()

            class_train_membership_probs = mia._compute_membership_probability(class_input_slice).train_membership_probs

            class_high_risk_idx = np.argsort(class_train_membership_probs)[::-1][:num_images]
            class_low_risk_idx = np.argsort(np.absolute(class_train_membership_probs - 0.5))[:num_images]

            high_risk_images = self.train_images[class_dataset_idx[class_high_risk_idx]]
            low_risk_images = self.train_images[class_dataset_idx[class_low_risk_idx]]

            fig = plt.figure(figsize=(10, 10 * num_images))
            for i in range(num_images):
                fig.add_subplot(1, num_images, i + 1)
                plt.axis("off")
                plt.imshow(high_risk_images[i])
            plt.savefig(f"high_risk_images_class_{c}.png")

            fig = plt.figure(figsize=(10, 10 * num_images))
            for i in range(num_images):
                fig.add_subplot(1, num_images, i + 1)
                plt.axis("off")
                plt.imshow(low_risk_images[i])
            plt.savefig(f"low_risk_images_class_{c}.png")


class AmiaAttack():
    """Implementation for multi class advanced mia attack.

    Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used


    Code mostly copied from: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/advanced_mia_example.py
    """

    def __init__(self, model: CNNModel, dataset: AbstractDataset, shadow_model_dir: str = "data/models/amia/shadow_models", num_shadow_models: int = 10, num_classes: Optional[int] = None):
        """Initialize MiaAttack class.

        Paramter:
        ---------
        model : CNNModel
        dataset : AbstractDataset - When instatiating a Dataset, a validation dataset is not needed, instead increase the train sice when specifying the train_val_test_split
        num_classes : int | None - needed for visualizing the membership Probability
        """
        self.cnn_model: CNNModel = model

        if dataset.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            sys.exit(1)

        if dataset.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            sys.exit(1)

        self.ds = dataset
        if self.ds.ds_info is not None and "classes" in self.ds.ds_info.keys():
            self.ds_classes: int = self.ds.ds_info["classes"]
        else:
            if num_classes is not None:
                self.ds_classes = num_classes
            else:
                print("ERROR: Number of classes was not specified by either the dataset nor the the classe's initialization")
                sys.exit(1)

        self.num_shadow_models = num_shadow_models
        self.ds_classes: int = 0
        self.models_dir = shadow_model_dir

        self.in_indices = []  # a list of in-training indices for all models
        self.stat = []  # a list of statistics for all models
        self.losses = []  # a list of losses for all models
        self.num_training_samples: int = 0
        self.sample_weight = None

    def train_load_shadow_models(self):
        """Trains, or if shadow models are already trained and saved, loads shadow models from filesystem.

        After training/ loading the shadow models statistics and losses are calulcated over all shadow models.

        """
        if not os.path.exists(self.models_dir):
            print(f"Creating directory: {self.models_dir}")
            os.makedirs(self.models_dir)

        # Train the target and shadow models. We will use one of the model in `models`
        # as target and the rest as shadow.
        # Here we use the same architecture and optimizer. In practice, they might
        # differ between the target and shadow models.

        train_values: np.ndarray = self.ds.get_train_values()
        train_labels: np.ndarray = self.ds.get_train_labels()

        self.num_training_samples = len(train_values)

        # we currently use the shadow and training models
        for i in range(self.num_shadow_models + 1):
            print(f"Creating shadow model {i}")

            model_path = os.path.join(self.models_dir,
                                      f"cnn_model_{i}_lr{self.cnn_model.learning_rate}_b{self.cnn_model.batch_size}_e{self.cnn_model.epochs}")

            # Generate a binary array indicating which example to include for training
            self.in_indices.append(np.random.binomial(1, 0.5, self.num_training_samples).astype(bool))

            # we want to create an exact copy of the already trained model, but change model path
            shadow_model: CNNModel = copy.copy(self.cnn_model)
            shadow_model.model_path = model_path
            shadow_model.reset_model_optimizer()

            # create Datasets for each shadow model based on the randomly selected training data
            train_value_slice = train_values[self.in_indices[-1]]
            train_label_slice = train_labels[self.in_indices[-1]]
            train_ds = tf.data.Dataset.from_tensor_slices((train_value_slice, train_label_slice))

            train_ds = train_ds.batch(batch_size=shadow_model.batch_size)

            val_value_slice = train_values[~self.in_indices[-1]]
            val_label_slice = train_labels[~self.in_indices[-1]]
            val_ds = tf.data.Dataset.from_tensor_slices((val_value_slice, val_label_slice))

            val_ds = val_ds.batch(batch_size=shadow_model.batch_size)

            # load model if already trained, else train & save it
            if os.path.exists(model_path):
                shadow_model.load_model()
                print(f"Loaded model {model_path} from disk")
            else:
                shadow_model.build_compile()
                shadow_model.train_model(train_ds, val_ds)
                shadow_model.save_model()
                print(f"Trained and saved model: {model_path}")

            stat_temp, loss_temp = self._get_stat_and_loss_aug(shadow_model, train_values, train_labels, sample_weight=self.sample_weight)
            self.stat.append(stat_temp)
            self.losses.append(loss_temp)

            # avoid OOM
            tf.keras.backend.clear_session()
            gc.collect()

    def attack_shadow_models_mia(self, plot_auc_curve: bool = True, plot_filename: Optional[str] = "advanced_mia.png"):
        print("Attacking shadow models with MIA")

        if len(self.stat) == 0 or len(self.losses) == 0:
            print("Error: Before attacking the shadow models with MIA, please train or load the shadow models and retrieve the statistics and losses")
            sys.exit(1)

        # we currently use the shadow and training models
        for idx in range(self.num_shadow_models + 1):
            print(f"Target model is #{idx}")
            stat_target = self.stat[idx]  # statistics of target model, shape(n,k)
            in_indices_target = self.in_indices[idx]  # ground truth membership, shape(n,)

            # `stat_shadow` contains statistics of the shadow models, with shape
            # (num_shadows, n, k). `in_indices_shadow` contains membership of the shadow
            # models, with shape (num_shadows, n). We will use them to get a list
            # `stat_in` and a list `stat_out`, where stat_in[j] (resp. stat_out[j]) is a
            # (m, k) array, for m being the number of shadow models trained with
            # (resp. without) the j-th example, and k being the number of augmentations
            # (2 in our case).
            stat_shadow = np.array(self.stat[:idx] + self.stat[idx + 1:])
            in_indices_shadow = np.array(self.in_indices[:idx] + self.in_indices[idx + 1:])
            stat_in = [stat_shadow[:, j][in_indices_shadow[:, j]] for j in range(self.num_training_samples)]
            stat_out = [stat_shadow[:, j][~in_indices_shadow[:, j]] for j in range(self.num_training_samples)]

            # compute the scores and use them for  MIA
            scores = amia.compute_score_lira(stat_target, stat_in, stat_out, fix_variance=True)

            attack_input = AttackInputData(
                loss_train=scores[in_indices_target],
                loss_test=scores[~in_indices_target],
                sample_weight_train=self.sample_weight,
                sample_weight_test=self.sample_weight)

            result_lira = mia.run_attacks(attack_input).single_attack_results[0]
            print("Advanced MIA attack with Gaussian:",
                  f"auc = {result_lira.get_auc():.4f}",
                  f"adv = {result_lira.get_attacker_advantage():.4f}")

            # We also try using `compute_score_offset` to compute the score. We take
            # the negative of the score, because higher statistics corresponds to higher
            # probability for in-training, which is the opposite of loss.
            scores = -amia.compute_score_offset(stat_target, stat_in, stat_out)
            attack_input = AttackInputData(
                loss_train=scores[in_indices_target],
                loss_test=scores[~in_indices_target],
                sample_weight_train=self.sample_weight,
                sample_weight_test=self.sample_weight)
            result_offset = mia.run_attacks(attack_input).single_attack_results[0]
            print('Advanced MIA attack with offset:',
                  f'auc = {result_offset.get_auc():.4f}',
                  f'adv = {result_offset.get_attacker_advantage():.4f}')

            # Compare with the baseline MIA using the loss of the target model
            loss_target = self.losses[idx][:, 0]
            attack_input = AttackInputData(
                loss_train=loss_target[in_indices_target],
                loss_test=loss_target[~in_indices_target],
                sample_weight_train=self.sample_weight,
                sample_weight_test=self.sample_weight)
            result_baseline = mia.run_attacks(attack_input).single_attack_results[0]
            print('Baseline MIA attack:',
                  f'auc = {result_baseline.get_auc():.4f}',
                  f'adv = {result_baseline.get_attacker_advantage():.4f}')

            if plot_auc_curve:
                print("Generating AUC curve plot")
                # Plot and save the AUC curves for the three methods.
                _, ax = plt.subplots(1, 1, figsize=(5, 5))
                for res, title in zip([result_baseline, result_lira, result_offset],
                                      ['baseline', 'LiRA', 'offset']):
                    label = f'{title} auc={res.get_auc():.4f}'
                    plotting.plot_roc_curve(
                        res.roc_curve,
                        functools.partial(self._plot_curve_with_area, ax=ax, label=label))
                plt.legend()
                plt.savefig(plot_filename)

    def _get_stat_and_loss_aug(self,
                               cnn_model: CNNModel,
                               x: np.ndarray,
                               y: np.ndarray,
                               sample_weight: Optional[np.ndarray] = None):
        """Get the statistics and losses.

        Paramter
        --------
          model: model to make prediction
          x: samples
          y: true labels of samples (integer valued)
          sample_weight: a vector of weights of shape (n_samples, ) that are
            assigned to individual samples. If not provided, then each sample is
            given unit weight. Only the LogisticRegressionAttacker and the
            RandomForestAttacker support sample weights.


        Note:   in the original code a batch_size is specified for the predict function,
                however since we work on 'dataset' we don't need it according to the documentation

        Returns
        -------
          the statistics and cross-entropy losses

        """
        losses, stat = [], []
        for data in [x, x[:, :, ::-1, :]]:
            prob = amia.convert_logit_to_prob(
                cnn_model.model.predict(data))
            losses.append(utils.log_loss(y, prob, sample_weight=sample_weight))
            stat.append(
                amia.calculate_statistic(
                    prob, y, sample_weight=sample_weight))
        return np.vstack(stat).transpose(1, 0), np.vstack(losses).transpose(1, 0)

    def _plot_curve_with_area(self, x, y, xlabel, ylabel, ax, label, title=None):
        ax.plot([0, 1], [0, 1], 'k-', lw=1.0)
        ax.plot(x, y, lw=2, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set(aspect=1, xscale='log', yscale='log')
        ax.title.set_text(title)
