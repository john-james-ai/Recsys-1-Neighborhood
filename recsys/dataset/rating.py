#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/rating.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 12:40:31 am                                               #
# Modified   : Wednesday March 1st 2023 12:03:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np
import pandas as pd
from typing import Union

from recsys.assets.dataset import Dataset


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset(Dataset):
    """Dataset containing movielens ratings data"""

    def __init__(
        self,
        name: str,
        description: str,
        data: pd.DataFrame,
        datasource: str = "movielens25m",
    ) -> None:
        super().__init__(name=name, description=description, data=data, datasource=datasource)

    @property
    def sparsity(self) -> float:
        """Returns measure of sparsity of the data in percent"""
        return self._sparsity

    @property
    def density(self) -> float:
        """Returns measure of density of the data in percent"""
        return self._density

    @property
    def n_users(self) -> int:
        """Returns number of unique users"""
        return self._n_users

    @property
    def n_items(self) -> int:
        """Returns number of unique items."""
        return self._n_items

    @property
    def utility_matrix_size(self) -> int:
        return self._utility_matrix_size

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

    def _run_profile(self) -> None:
        """Runs a data profile including basic summary statistics"""
        if not self._profiled:
            super()._run_profile()
            # Computes basic statistics
            self._n_users = int(self._data["userId"].nunique())
            self._n_items = int(self._data["movieId"].nunique())
            self._utility_matrix_size = int(self._n_users * self._n_items)
            self._sparsity = self._nrows / self._size * 100
            self._density = 100 - self._sparsity
            self._memory = self._data.memory_usage(deep=True).sum()

            d = {}
            d["id"] = self._id
            d["name"] = self._name
            d["type"] = self.__class__.__name__
            d["description"] = self._description
            d["workspace"] = self._workspace
            d["nrows"] = self._data.shape[0]
            d["ncols"] = self._data.shape[1]
            d["n_users"] = self._n_users
            d["n_items"] = self._n_items
            d["size"] = self._size
            d["utility_matrix_size"] = self._utility_matrix_size
            d["memory"] = self._memory
            d["sparsity"] = self._sparsity
            d["density"] = self._density

            self._profile = pd.DataFrame.from_dict(data=d, orient="index", columns=["Metric"])

        self._profiled = True
