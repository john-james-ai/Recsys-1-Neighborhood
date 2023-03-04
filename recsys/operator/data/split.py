#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/split.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 09:20:09 pm                                               #
# Modified   : Friday March 3rd 2023 03:40:37 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Train/Test Split Module"""
import os

import pandas as pd

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                                TEMPORAL TRAIN/TEST SPLIT                                         #
# ------------------------------------------------------------------------------------------------ #
class TemporalTrainTestSplit(Operator):
    """Temporal train test split uses timestamp to split along a temporal dimension

    Args:
        source (str): Optional the
        train_size (float): The proportion of data to allocate to the training set.
        timestamp_var (str): The variable containing the timestamp
        source (str): Source file path. Optional
        destination (str): The output directory. Optional)
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        train_size: float = 0.8,
        timestamp_var: str = "timestamp",
        source: str = None,
        destination: str = None,
        train_filename: str = "train.pkl",
        test_filename: str = "test.pkl",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._train_filename = train_filename
        self._test_filename = test_filename
        self._train_filepath = os.path.join(self._destination, self._train_filename)
        self._test_filepath = os.path.join(self._destination, self._test_filename)
        self._train_size = train_size
        self._timestamp_var = timestamp_var

    def execute(self, data: pd.DataFrame = None) -> None:
        """Performs the train test split."""
        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:

                data_sorted = data.sort_values(by=[self._timestamp_var], ascending=True)
                train_size = int(self._train_size * data.shape[0])

                train = data_sorted[0:train_size]
                test = data_sorted[train_size:]

                self._put_data(filepath=self._train_filepath, data=train)
                self._put_data(filepath=self._test_filepath, data=test)

                result = {"train": train, "test": test}
                return result
            except KeyError:
                msg = f"Column {self._timestamp_var} was not found."
                self._logger.error(msg)
                raise
