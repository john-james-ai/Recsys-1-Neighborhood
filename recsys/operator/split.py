#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/split.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 09:20:09 pm                                               #
# Modified   : Friday February 24th 2023 11:12:11 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Train/Test Split Module"""
import os

from recsys.operator.base import Operator


# ------------------------------------------------------------------------------------------------ #
#                                TEMPORAL TRAIN/TEST SPLIT                                         #
# ------------------------------------------------------------------------------------------------ #
class TemporalTrainTestSplit(Operator):
    """Temporal train test split uses timestamp to split along a temporal dimension

    Args:
        train_size (float): The proportion of data to allocate to the training set.
        source (str): Filepath of the source file
        destination (str): Directory into which the training and test files will be saved.
        train_filename (str): The filename of the train file
        test_filename (str): The filename of the test file.
        timestamp_var (str): The variable containing the timestamp
        force (bool): Whether to force execution
    """

    def __init__(
        self,
        source: str,
        destination: str,
        train_filename: str,
        test_filename: str,
        train_size: float = 0.8,
        timestamp_var: str = "timestamp",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._train_filename = train_filename
        self._test_filename = test_filename
        self._train_filepath = os.path.join(self._destination, self._train_filename)
        self._test_filepath = os.path.join(self._destination, self._test_filename)
        self._train_size = train_size
        self._timestamp_var = timestamp_var

    def run(self) -> None:
        """Performs the train test split."""
        data = self._fio.read()
        try:
            data_sorted = data.sort_values(by=[self._timestamp_var], ascending=True)
        except KeyError:
            msg = f"Column {self._timestamp_var} was not found."
            self._logger.error(msg)
            raise ValueError(msg)

        train_size = int(self._train_size * len(data_sorted))

        train = data_sorted[0:train_size]
        test = data_sorted[train_size:]

        self._fio.write(filepath=self._train_filepath, data=train)
        self._fio.write(filepath=self._test_filepath, data=test)
