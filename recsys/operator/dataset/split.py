#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/dataset/split.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 09:20:09 pm                                               #
# Modified   : Friday March 17th 2023 03:00:23 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Train/Test Split Module"""
import os

import pandas as pd

from recsys import Operator, Artifact


# ------------------------------------------------------------------------------------------------ #
#                                TEMPORAL TRAIN/TEST SPLIT                                         #
# ------------------------------------------------------------------------------------------------ #
class TemporalTrainTestSplit(Operator):
    """Temporal train test split uses timestamp to split along a temporal dimension

    Args:
        source (str): Source file path.
        train_filepath (str): The path to the train file.
        test_filepath (str): The path to the test file.
        train_size (float): The proportion of data to allocate to the training set.
        timestamp_var (str): The variable containing the timestamp
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        source: str,
        train_filepath: str,
        test_filepath: str,
        train_size: float = 0.8,
        timestamp_var: str = "timestamp",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, force=force)
        self._train_filepath = train_filepath
        self._test_filepath = test_filepath
        self._train_size = train_size
        self._timestamp_var = timestamp_var
        directory = os.path.dirname(self._train_filepath)
        self._artifact = Artifact(isfile=False, path=directory, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
        """Performs the train test split."""
        if not self._skip(endpoint=self._train_filepath) or not self._skip(
            endpoint=self._test_filepath
        ):

            data = self._get_data(filepath=self._source)

            try:

                data_sorted = data.sort_values(by=[self._timestamp_var], ascending=True)
                train_size = int(self._train_size * data.shape[0])

                train = data_sorted[0:train_size]
                test = data_sorted[train_size:]

                self._put_data(filepath=self._train_filepath, data=train)
                self._put_data(filepath=self._test_filepath, data=test)

                self._logger.info(f"Created training set at {self._train_filepath}.")
                self._logger.info(f"Created test set at {self._test_filepath}.")

            except KeyError:
                msg = f"Column {self._timestamp_var} was not found."
                self._logger.error(msg)
                raise
