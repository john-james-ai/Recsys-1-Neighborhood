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
# Modified   : Saturday February 4th 2023 06:32:47 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Exploratory Data Analysis Module"""
import os
import pandas as pd
import numpy as np
from typing import Union
from copy import copy
from scipy import sparse

from recsys.io.file import IOService


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset:
    """Ratings class

    Args:
        filepath (str): Path to ratings data
    """

    def __init__(self, filepath: str) -> None:
        self._name = os.path.basename(filepath)
        self._filepath = filepath
        self._data = IOService.read(filepath)
        self._csr = None
        self._ave_user_ratings = None
        self._ave_item_ratings = None
        self._user_id_map = None
        self._item_id_map = None
        self._remap_ids()

    def __len__(self) -> int:
        """Returns number of rows in the dataset"""
        return len(self._data)

    def __eq__(self, other) -> bool:
        """Determines equality with other object"""
        if isinstance(other, self.__class__):
            return (
                self.name == other.name
                and self.nrows == other.nrows
                and self.ncols == other.ncols
                and self.size == other.size
                and np.array_equal(self.users, other.users)
                and np.array_equal(self.items, other.items)
            )
        else:
            return False

    def info(self) -> None:
        """Wraps pandas info method."""
        self._data.info()

    @property
    def name(self) -> str:
        return self._name

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return self._data.shape[0]

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return self._data.shape[1]

    @property
    def memory(self) -> float:
        """Returns memory size of dataset in Mb"""
        return self._data.memory_usage(deep=True).sum() / (1024**2)

    @property
    def size(self) -> int:
        return self.nrows * self.ncols

    @property
    def matrix_size(self) -> int:
        return self.n_users * self.n_items

    @property
    def sparsity(self) -> float:
        """Returns measure of sparsity of the data in percent"""
        return self.size / self.matrix_size * 100

    @property
    def density(self) -> float:
        """Returns measure of density of the data in percent"""
        return 1 - self.sparsity

    @property
    def users(self) -> np.array:
        return np.sort(self._data["userId"].unique())

    @property
    def items(self) -> np.array:
        return np.sort(self._data["movieId"].unique())

    @property
    def n_users(self) -> int:
        """Returns number of unique users"""
        return self._data["userId"].nunique()

    @property
    def n_items(self) -> int:
        """Returns number of unique items."""
        return self._data["movieId"].nunique()

    def summarize(self) -> pd.DataFrame:
        """Returns a data frame with summary statistics"""
        d = {
            "Rows": self.nrows,
            "Columns": self.ncols,
            "Users": self.n_users,
            "Movies": self.n_items,
            "Memory Usage": self.memory,
            "Size": self.size,
            "Matrix Size": self.matrix_size,
            "Sparsity": self.sparsity,
            "Density": self.density,
            "Maximum Number of Ratings by User": self._data["userId"].value_counts().max(),
            "Average Number of Ratings by User": self._data["userId"].value_counts().sum()
            / self.n_users,
            "Maximum Number of Ratings for Movie": self._data["movieId"].value_counts().max(),
            "Average Number of Ratings for Movie": self._data["movieId"].value_counts().sum()
            / self.n_items,
        }
        df = pd.DataFrame.from_dict(data=d, orient="index", columns=["Count"])
        return df

    def as_dataframe(self) -> None:
        """Returns a copy of the ratings dataframe."""
        return copy(self._data)

    def as_csr(self) -> sparse.csr_matrix:
        """Returns ratings data in csr sparse format"""
        if self._csr is None:
            row = self._data.useridx.values
            col = self._data.itemidx.values
            data = self._data.rating.values
            self._csr = sparse.csr_matrix(
                (data, (row, col)),
                shape=(self._data.useridx.nunique(), self._data.itemidx.nunique()),
            )
        return self._csr.copy()

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

    def get_items(self, userId: int) -> np.array:
        """Returns an array of items rated by the designated user

        Args:
            userId (str): Id for the user

        Returns: np.array
        """
        return np.sort(self._data[self._data["userId"] == userId]["movieId"].values)

    def get_users(self, movieId: int) -> np.array:
        """Returns an array of users which rated the designated movie

        Args:
            movieId (str): Id for the movie

        Returns: np.array
        """
        return np.sort(self._data[self._data["movieId"] == movieId]["userId"].values)

    def get_users_items_ratings(self, items: list, users: list) -> pd.DataFrame:
        """Subsets ratings by users and items.

        The original ratings are returned, unless normalized by is set to 'user', or 'item'; whereby,
        the ratings are normalized by the user average rating or the item average rating, respectively.

        Args:
            items (List[int]): List of ids for items of interest
            users (List[int]): List of ids for users of interest

        """
        return self._data[(self._data["userId"].isin(users)) & (self._data["movieId"].isin(items))]

    def get_user_ratings(self, user: int) -> pd.DataFrame:
        """Gets all user ratings.

        Args:
            user (int): The id for the user for whom the ratings are being returned.
        """

        return self._data[self._data["userId"] == user].sort_values(
            by="movieId", ascending=True, axis=0
        )

    def get_item_ratings(self, item: int) -> pd.DataFrame:
        """Gets all item ratings.

        Args:
            item (int): The id for item for which the ratings are being returned.

        """

        return self._data[self._data["movieId"] == item].sort_values(
            by="userId", ascending=True, axis=0
        )

    def get_ave_user_ratings(self, user: int = None) -> Union[float, pd.DataFrame]:
        """Returns user average ratings

        For a user, the method returns a float. If user is not specified, a DataFrame
        of all user average ratings will be returned.

        Args:
            user (int): Optional. User for which average ratings are being requested.
        """
        if self._ave_user_ratings is None:
            self._ave_user_ratings = self._data.groupby("userId")["rating"].mean().reset_index()
            self._ave_user_ratings.columns = ["userId", "rbar"]

        if user is None:
            return self._ave_user_ratings
        else:
            return self._ave_user_ratings[self._ave_user_ratings["userId"] == user]["rbar"].values[
                0
            ]

    def get_ave_item_ratings(self, item: int = None) -> Union[float, pd.DataFrame]:
        """Returns item average ratings

        For an item, the method returns a float. If item is not specified, a DataFrame
        of all item average ratings will be returned.

        Args:
            item (int): Optional. Item id for which average ratings are being requested.
        """
        if self._ave_item_ratings is None:
            self._ave_item_ratings = self._data.groupby("movieId")["rating"].mean().reset_index()
            self._ave_item_ratings.columns = ["movieId", "rbar"]

        if item is None:
            return self._ave_item_ratings
        else:
            return self._ave_item_ratings[self._ave_item_ratings["movieId"] == item]["rbar"].values[
                0
            ]

    def _remap_ids(self) -> None:
        """Remaps the ids to sequential range."""
        # Create User Map
        userId = np.sort(self._data["userId"].unique())
        useridx = range(len(userId))
        u = {"userId": userId, "useridx": useridx}
        self._user_id_map = pd.DataFrame(data=u)

        # Create Item Map
        movieId = np.sort(self._data["movieId"].unique())
        itemidx = range(len(movieId))
        i = {"movieId": movieId, "itemidx": itemidx}
        self._item_id_map = pd.DataFrame(data=i)

        # Install New Indices
        self._data = self._data.merge(self._user_id_map, on="userId", how="left")
        self._data = self._data.merge(self._item_id_map, on="movieId", how="left")
