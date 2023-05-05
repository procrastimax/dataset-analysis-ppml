import tensorflow as tf
from scipy import special
from typing import Optional
from model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
import numpy as np


from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, SlicingSpec, AttackType, SingleSliceSpec, SlicingFeature
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.dataset_slicing import get_slice
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia

import matplotlib.pyplot as plt

from os import sys


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
                print(
                    "ERROR: Number of classes was not specified by either the dataset nor the the classe's initialization")
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
        self.logits_train = self.cnn_model.model.predict(
            self.ds.ds_attack_train, batch_size=self.cnn_model.batch_size)

        print("Predict on unseen test data...")
        self.logits_test = self.cnn_model.model.predict(
            self.ds.ds_attack_test, batch_size=self.cnn_model.batch_size)

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
        self.loss_train = self.scce(self.constant(self.train_labels),
                                    self.constant(self.prob_train), from_logits=False).numpy()
        self.loss_test = self.scce(self.constant(self.test_labels),
                                   self.constant(self.prob_test), from_logits=False).numpy()

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
                print(
                    "ERROR: Number of classes was not specified by either the dataset nor the the classe's initialization")
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
        self.logits_train = self.cnn_model.model.predict(
            self.ds.ds_attack_train, batch_size=self.cnn_model.batch_size)

        print("Predict on unseen test data...")
        self.logits_test = self.cnn_model.model.predict(
            self.ds.ds_attack_test, batch_size=self.cnn_model.batch_size)

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
        self.loss_train = self.scce(self.constant(self.train_labels),
                                    self.constant(self.prob_train), from_logits=False).numpy()
        self.loss_test = self.scce(self.constant(self.test_labels),
                                   self.constant(self.prob_test), from_logits=False).numpy()

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

        membership_probability_results = mia.run_membership_probability_analysis(
            self.input, slicing_spec=slicing_spec)
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

            class_train_membership_probs = mia._compute_membership_probability(
                class_input_slice).train_membership_probs

            class_high_risk_idx = np.argsort(class_train_membership_probs)[::-1][:num_images]
            class_low_risk_idx = np.argsort(np.absolute(
                class_train_membership_probs - 0.5))[:num_images]

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
