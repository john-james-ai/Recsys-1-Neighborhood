#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/entity/rating.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 07:10:48 pm                                             #
# Modified   : Monday February 20th 2023 11:29:47 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np
import pandas as pd
from typing import Union

from recsys.entity.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset(Dataset):
    """Dataset containing movielens ratings data"""

    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        description: str,
        stage: str = "staged",
    ) -> None:
        super().__init__(name=name, description=description, stage=stage)
        self._data = data
        self._profiled = False
        self.profile()

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return self._nrows

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return self._ncols

    @property
    def nnz(self) -> int:
        """Returns number cells in the dataframe."""
        return self._nnz

    @property
    def size(self) -> int:
        """Returns number of elements in a user / item matrix"""
        return self._size

    @property
    def sparsity(self) -> float:
        """Returns measure of sparsity of the data in percent"""
        return self._sparsity

    @property
    def density(self) -> float:
        """Returns measure of density of the data in percent"""
        return self._density

    @property
    def memory(self) -> int:
        return self._memory

    @property
    def n_users(self) -> int:
        """Returns number of unique users"""
        return self._n_users

    @property
    def n_items(self) -> int:
        """Returns number of unique items."""
        return self._n_items

    @property
    def users(self) -> np.array:
        """Returns array of unique users"""
        return np.sort(self._data["userId"].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data["movieId"].unique())

    @property
    def user_item_ratio(self) -> float:
        """Returns ratio of the number of users to items."""
        return self.n_users / self.n_items

    @property
    def item_user_ratio(self) -> float:
        """Returns ratio of the number of items to users."""
        return self.n_items / self.n_users

    @property
    def user_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by user."""
        return (
            self._data["userId"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["userId"])
        )

    @property
    def user_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of user rating frequency"""
        return self.user_rating_counts.describe().T

    @property
    def item_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by item."""
        return (
            self._data["movieId"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["movieId"])
        )

    @property
    def item_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of item rating frequency"""
        return self.item_rating_counts.describe().T

    def get_user_ratings(
        self, userId: int, mean_centered: Union[str, bool] = False
    ) -> pd.DataFrame:
        """Returns ratings created by user.
        Args:
            userId (int): index for the user
        Returns: pd.DataFrame
        """
        if mean_centered is not False:
            data = self._extract_data(mean_centered=mean_centered)
        else:
            data = self._data
        return data[data["userId"] == userId]

    def get_item_ratings(
        self, movieId: int, mean_centered: Union[str, bool] = False
    ) -> pd.DataFrame:
        """Returns ratings for the given item
        Args:
            movieId (int): Index for the item / movie
        Returns: pd.DataFrame
        """
        if mean_centered is not False:
            data = self._extract_data(mean_centered=mean_centered)
        else:
            data = self._data
        return data[data["movieId"] == movieId]

    def get_users_rated_item(self, movieId: int) -> list:
        """Returns a list of users who have rated movieId
        Args:
            movieId (int): The index for the item
        """
        return self._data[self._data["movieId"] == movieId]["userId"].tolist()

    def get_items_rated_user(self, userId: int) -> list:
        """Returns a list of items rated by userId.
        Args:
            userId (int): The index for the user
        """
        return self._data[self._data["userId"] == userId]["movieId"].tolist()

    def profile(self) -> pd.DataFrame:
        """Returns a data profile including basic summary statistics"""
        if not self._profiled:
            # Computes basic statistics
            self._nrows = int(self._data.shape[0])
            self._ncols = int(self._data.shape[1])
            self._n_users = int(self._data["userId"].nunique())
            self._n_items = int(self._data["movieId"].nunique())
            self._nnz = int(self._nrows * self._ncols)
            self._size = int(self._n_users * self._n_items)
            self._sparsity = self._nrows / self._size * 100
            self._density = 100 - self._sparsity
            self._memory = self._data.memory_usage(deep=True).sum()

            d = {}
            d["name"] = self._name
            d["type"] = self._type
            d["description"] = self._description
            d["workspace"] = self._workspace
            d["stage"] = self._stage
            d["nrows"] = self._data.shape[0]
            d["ncols"] = self._data.shape[1]
            d["n_users"] = self._n_users
            d["n_items"] = self._n_items
            d["nnz"] = self._nnz
            d["size"] = self._matrix_size
            d["memory"] = self._memory
            d["sparsity"] = self._sparsity
            d["density"] = self._density

            self._df = pd.DataFrame.from_dict(data=d, orient="index", columns=["Count"])

        self._profiled = True
        return self._df

    def as_df(self) -> pd.DataFrame:
        """Returns a pandas dataframe with key metadata."""
        self.profile()
