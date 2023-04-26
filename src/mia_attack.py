import tensorflow as tf
from scipy import special
from model import CNNModel
from ppml_datasets.abstract_dataset_handler import AbstractDataset
import numpy as np

from typing import List

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia


class MiaAttack():
    """Implementation for multi class mia attack.
       Labels are encoded as multi-class labels, so no one-hot encoding -> a sparse_categorical_crossentropy is used
    """

    def __init__(self, model: CNNModel, dataset: AbstractDataset):
        self.cnn_model = model
        self.ds = dataset

        if dataset.ds_train is None:
            print("Error: Dataset needs to have an initialized train dataset!")
            return

        if dataset.ds_test is None:
            print("Error: Dataset needs to have an initialized test dataset!")
            return

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

    def _get_labels_from_ds(self, ds: tf.data.Dataset) -> np.array:
        # labels
        labels = []
        for x, y in ds.as_numpy_iterator():
            labels.append(y[0])

        return np.asarray(labels)

    def initialize_data(self):
        """Initialize and calculate logits, probabilities and loss values for training and test sets."""
        print("Predict on train data...")
        self.logits_train = self.cnn_model.model.predict(self.ds.ds_attack_train, batch_size=self.cnn_model.batch_size)

        print("Predict on unseen test data...")
        self.logits_test = self.cnn_model.model.predict(self.ds.ds_attack_test, batch_size=self.cnn_model.batch_size)

        print("Apply softmax to get probabilities from logits")
        self.prob_train = special.softmax(self.logits_train, axis=1)
        self.prob_test = special.softmax(self.logits_test, axis=1)

        print("Get labels for datasets")
        self.train_labels = self._get_labels_from_ds(self.ds.ds_attack_train)
        self.test_labels = self._get_labels_from_ds(self.ds.ds_attack_test)

        print("Compute losses")
        self.loss_train = self.scce(self.constant(self.train_labels), self.constant(self.prob_train), from_logits=False).numpy()
        self.loss_test = self.scce(self.constant(self.test_labels), self.constant(self.prob_test), from_logits=False).numpy()

    def run_mia_attack(self):
        print("Running MIA attacks")

        # Suppose we have the labels as integers starting from 0
        # labels_train  shape: (n_train, )
        # labels_test  shape: (n_test, )

        # Evaluate your model on training and test examples to get
        # logits_train  shape: (n_train, n_classes)
        # logits_test  shape: (n_test, n_classes)
        # loss_train  shape: (n_train, )
        # loss_test  shape: (n_test, )

        input = AttackInputData(
            logits_train=self.logits_train,
            logits_test=self.logits_test,
            loss_train=self.loss_train,
            loss_test=self.loss_test,
            labels_train=self.train_labels,
            labels_test=self.test_labels
        )

        # run attacks for different data slices
        attacks_result = mia.run_attacks(input,
                                         SlicingSpec(entire_dataset=True,
                                                     by_class=True,
                                                     by_classification_correctness=True),
                                         attack_types=[AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION])

        # Print a user-friendly summary of the attacks
        print(attacks_result.summary(by_slices=True))

        # Plot the ROC curve of the best classifier
        fig = plotting.plot_roc_curve(
            attacks_result.get_result_with_max_auc().roc_curve)
        fig.savefig("mia_attcks.png")
