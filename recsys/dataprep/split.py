#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataprep/split.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 09:20:09 pm                                               #
# Modified   : Monday March 20th 2023 09:50:08 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Train/Test Split Module"""
from __future__ import annotations
import os
import logging

from recsys import Operator
from recsys.dataprep.artifact import Artifact
from recsys.services.log import log
from recsys.dataset.base import Dataset
from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
#                                TEMPORAL TRAIN/TEST SPLIT                                         #
# ------------------------------------------------------------------------------------------------ #
class TemporalSplitOperator(Operator):
    """Temporal train validatioon, and test split uses timestamp to split along a temporal dimension

    Args:
        directory (str): The directory into which the splits will be persisted.
        train_size (float): Proportion of data for training. Default = 0.80
        test_size (float): Proportion of data for test. Default = 0.1
        validation_size (float): Proportion reserved for validation set. This will equal
            1 - (train_size + test_size)
        timestamp_var (str): The variable containing the timestamp
        force (bool): Whether to overwrite existing data if it already exists.
    """

    __name = "temporal_split_operator"
    __desc = "Splits datasets by fixed points in time."

    def __init__(
        self,
        directory: str,
        train_size: float = 0.80,
        validation_size: float = 0.10,
        test_size: float = 0.10,
        timestamp_var: str = "timestamp",
        force: bool = False,
    ) -> None:
        super().__init__()
        self._directory = directory
        self._train_size = train_size
        self._validation_size = validation_size or 1 - (train_size + test_size)
        self._test_size = test_size
        self._timestamp_var = timestamp_var
        self._artifact = Artifact(isfile=False, path=directory, uripath="data")
        self._force = force
        self._validate()
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @log
    def __call__(self, dataset: Dataset) -> None:
        """Performs the train (validation) and test split."""

        splits = {}  # Container of Dataset objects

        if not self._skip(endpoint=self._directory):

            data = dataset.to_df()

            try:
                data_sorted = data.sort_values(by=[self._timestamp_var], ascending=True)
            except KeyError:
                msg = "The timestamp variable is invalid."
                self._logger.error(msg)
                raise ValueError(msg)

            total_examples = data.shape[0]

            train_size = int(self._train_size * data.shape[0])
            test_size = int(self._test_size * data.shape[0])
            validation_size = min(
                int(self._validation_size * data.shape[0]), total_examples - train_size + test_size
            )

            train_begin = 0
            train_end = train_size
            validation_begin = train_end
            validation_end = validation_begin + validation_size
            test_begin = validation_end
            test_end = test_begin + test_size

            train = data_sorted[train_begin:train_end]
            validation = data_sorted[validation_begin:validation_end]
            test = data_sorted[test_begin:test_end]

            # Training set
            name = "train"
            desc = "Temporal Training Set"
            filepath = os.path.join(self._directory, "train.pkl")
            data = train
            dataset = Dataset(name=name, desc=desc, filepath=filepath, data=data)
            IOService.write(filepath=dataset.filepath, data=dataset)
            splits["train"] = dataset

            # Test set
            name = "test"
            desc = "Temporal Test Set"
            filepath = os.path.join(self._directory, "test.pkl")
            data = test
            dataset = Dataset(name=name, desc=desc, filepath=filepath, data=data)
            IOService.write(filepath=dataset.filepath, data=dataset)
            splits["test"] = dataset

            # Validation set
            if validation_size > 0.0:
                name = "validation"
                desc = "Temporal Validation Set"
                filepath = os.path.join(self._directory, "validation.pkl")
                data = validation
                dataset = Dataset(name=name, desc=desc, filepath=filepath, data=data)
                IOService.write(filepath=dataset.filepath, data=dataset)
                splits["validation"] = dataset

        else:
            train_filepath = os.path.join(self._directory, "train.pkl")
            train = IOService.read(filepath=train_filepath)
            splits["train"] = train

            test_filepath = os.path.join(self._directory, "test.pkl")
            test = IOService.read(filepath=test_filepath)
            splits["test"] = test

            validation_filepath = os.path.join(self._directory, "validation.pkl")
            if os.path.exists(validation_filepath):
                validation = IOService.read(filepath=validation_filepath)
                splits["validation"] = validation

        return splits

    def _validate(self) -> None:
        """Ensures the sizes sum to one."""
        total_size = self._train_size + self._validation_size + self._test_size
        if total_size != 1.0:
            msg = f"Training, validation and test sizes sum to {total_size}. The sizes must sum to one."
            self._logger.error(msg)
            raise ValueError(msg)

        if self._train_size == 0.0 or self._test_size == 0.0:
            msg = "Training and test set sizes must be greater than 0.0"
            self._logger.error(msg)
            raise ValueError(msg)
