#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/model/split.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 4th 2023 12:41:23 am                                              #
# Modified   : Saturday February 4th 2023 01:43:43 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import pandas as pd

from recsys.model.base import DataSplitter
from recsys.io.file import IOService


# ------------------------------------------------------------------------------------------------ #
class TemporalSplitter(DataSplitter):
    """Splits the data based upon temporal data."""

    def __init__(self) -> None:
        super().__init__()

    def split(
        self,
        data: pd.DataFrame,
        train_size: float,
        split_var: str,
        train_filepath: str = None,
        test_filepath: str = None,
    ) -> dict[pd.DataFrame]:
        """Splits and optionally stores the data

        Args:
            data (pd.DataFrame): Data to be split
            train_size (float): The proportion of the data to include in the training split
            split_var (str): The variable containing a timestamp or datetime value.
            train_filepath (str): Optional path to which the train set will be saved
            test_filepath (str): Optional path to which the test set will be saved
        """
        try:
            data_sorted = data.sort_values(by=[split_var], ascending=True)
        except KeyError:
            msg = f"Column {split_var} was not found."
            self._logger.error(msg)
            raise ValueError(msg)

        train_size = int(train_size * len(data_sorted))

        train = data_sorted[0:train_size]
        test = data_sorted[train_size:]

        if train_filepath is not None:
            self.save(data=train, filepath=train_filepath)
        if test_filepath is not None:
            self.save(data=test, filepath=test_filepath)

        return {"train": train, "test": test}

    def save(self, data: pd.DataFrame, filepath: str) -> None:
        """Saves the data to disk

        Args:
            data (pd.DataFrame): The data to save
            filepath (str): Path to which the data will be saved

        """

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        IOService.write(filepath=filepath, data=data)
