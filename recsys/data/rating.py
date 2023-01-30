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
# Modified   : Monday January 30th 2023 07:00:44 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Exploratory Data Analysis Module"""
import os
import pandas as pd
import numpy as np

from recsys.io.file import IOService


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset:
    """Ratings class

    Args:
        filepath (str): Path to ratings data
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._data = IOService.read(filepath)

    def __len__(self) -> int:
        """Returns number of rows in the dataset"""
        return len(self._data)

    def info(self) -> None:
        """Wraps pandas info method."""
        self._data.info()

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return self._data.shape[0]

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return self._data.shape[1]

    @property
    def size(self) -> float:
        """Returns memory size of dataset in Mb"""
        return self._data.memory_usage(deep=True).sum() / (1024**2)

    @property
    def users(self) -> int:
        """Returns number of unique users"""
        return self._data["userId"].nunique()

    @property
    def items(self) -> int:
        """Returns number of unique items."""
        return self._data["movieId"].nunique()

    def summarize(self) -> pd.DataFrame:
        """Returns a data frame with summary statistics"""
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
        """Creates training and test sets

        Args:
            train_filepath (str): Path for training set
            test_filepath (str): Path to test set
            train_prop (float): Proportion of data to allocate to train set. Test set
                allocation is 1-train_prop

        """
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
        """Returns and optionally stores sample from the dataset.

        Args:
            frac (float): Proportion of dataset to sample
            filepath (str): Optional path for saving the sample
            random_state (int): Seed for pseudo random sampling

        Returns: pd.DataFrame
        """
        df = self._data.sample(
            frac=frac, replace=False, ignore_index=True, random_state=random_state
        )
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            IOService.write(filepath=filepath, data=df)
        return df

    def top_n_users(self, n: int = 10) -> pd.DataFrame:
        """Returns the users with n highest number of ratings.

        Args:
            n (int): The number of top users to return

        Returns: pd.DataFrame with userId and rating counts.

        """
        return self._data["userId"].value_counts(sort=True).to_frame("Counts")[0:n]

    def top_n_items(self, n: int = 10) -> pd.DataFrame:
        """Returns the items with n highest number of ratings.

        Args:
            n (int): The number of top items to return

        Returns: pd.DataFrame with movieId and rating counts.

        """
        return self._data["movieId"].value_counts(sort=True).to_frame("Counts")[0:n]

    def get_items(self, userId: str) -> np.array:
        """Returns an array of items rated by the designated user

        Args:
            userId (str): Id for the user

        Returns: np.array
        """
        return self._data[self._data["userId"] == userId]["movieId"].values

    def get_users(self, movieId: str) -> np.array:
        """Returns an array of users which rated the designated movie

        Args:
            movieId (str): Id for the movie

        Returns: np.array
        """
        return self._data[self._data["movieId"] == movieId]["userId"].values
