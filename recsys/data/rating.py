#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/rating.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 05:10:07 am                                                #
# Modified   : Sunday January 29th 2023 08:42:36 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Exploratory Data Analysis Module"""
import os
import pandas as pd

from recsys.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class RatingDataset:
    """Ratings class

    Args:
        filepath (str): Path to ratings data
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._data = IOService.read(filepath)

    def __len__(self) -> int:
        return len(self._data)

    def info(self) -> None:
        self._data.info()

    @property
    def nrows(self) -> int:
        return self._data.shape[0]

    @property
    def ncols(self) -> int:
        return self._data.shape[1]

    @property
    def size(self) -> int:
        return self._data.memory_usage(deep=True).sum()

    @property
    def users(self) -> int:
        return self._data["userId"].nunique()

    @property
    def items(self) -> int:
        return self._data["movieId"].nunique()

    def summarize(self) -> pd.DataFrame:
        d = {
            "Size": self._data.memory_usage(deep=True).sum(),
            "Rows": self._data.shape[0],
            "Columns": self._data.shape[1],
            "Users": self._data["userId"].nunique(),
            "Movies": self._data["movieId"].nunique(),
            "Maximum Ratings per User": self._data["userId"].value_counts().max(),
            "Average Ratings per User": self._data["userId"].value_counts().mean(),
            "Median Ratings per User": self._data["userId"].value_counts().median(),
            "Minimum Ratings per User": self._data["userId"].value_counts().min(),
            "Maximum Ratings per Movie": self._data["movieId"].value_counts().max(),
            "Average Ratings per Movie": self._data["movieId"].value_counts().mean(),
            "Median Ratings per Movie": self._data["movieId"].value_counts().median(),
            "Minimum Ratings per Movie": self._data["movieId"].value_counts().min(),
        }
        df = pd.DataFrame.from_dict(data=d, orient="index", columns=["Count"])
        print(df)
        return df

    def split(self, train_filepath: str, test_filepath: str, train_prop: float = 0.8) -> None:
        data_sorted = self._data.sort_values(by=["timestamp"], ascending=True)
        train_size = int(train_prop * len(data_sorted))

        train = data_sorted[0:train_size]
        test = data_sorted[train_size:]

        os.makedirs(os.path.dirname(train_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(test_filepath), exist_ok=True)

        IOService.write(filepath=train_filepath, data=train)
        IOService.write(filepath=test_filepath, data=test)

    def sample(
        self, frac: float = 0.2, filepath: str = None, random_state: int = None
    ) -> pd.DataFrame:
        df = self._data.sample(
            frac=frac, replace=False, ignore_index=True, random_state=random_state
        )
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            IOService.write(filepath=filepath, data=df)
        return df

    def top_n_users(self, n: int = 10) -> pd.DataFrame:
        return self._data["userId"].value_counts(sort=True).to_frame("Counts")[0:n]

    def top_n_items(self, n: int = 10) -> pd.DataFrame:
        return self._data["movieId"].value_counts(sort=True).to_frame("Counts")[0:n]
