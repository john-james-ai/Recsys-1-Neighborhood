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
# Modified   : Friday February 17th 2023 05:29:31 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Exploratory Data Analysis Module"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Union
from copy import copy
from scipy import sparse
from collections import namedtuple

from recsys.io.file import IOService


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset:
    """Ratings class

    Args:
        filename (str): The name of the file containing ratings data
        mode (str): One of 'dev', 'prod', or 'test
    """

    def __init__(self, filename: str, mode: str = "dev") -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )
        self._filename = filename
        self._mode = mode
        self._name = self._mode + "_" + os.path.basename(self._filename)
        self._filepath = self._get_filepath()

        try:
            self._data = IOService.read(self._filepath)
        except FileNotFoundError:
            msg = f"File {self._filename} not found in {self._mode} mode."
            self._logger.error(msg)
            raise ValueError(msg)

        self._ave_user_ratings = None
        self._ave_item_ratings = None
        self._user_id_map = None
        self._item_id_map = None
        self._centered = False
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

    def reset(self) -> None:
        """Returns dataset to state at construction."""
        self._data = IOService.read(self._filepath)
        self._csr = {}
        self._csc = {}
        self._ave_user_ratings = None
        self._ave_item_ratings = None
        self._user_id_map = None
        self._item_id_map = None
        self._centered = False
        self._remap_ids()

    def info(self) -> None:
        """Wraps pandas info method."""
        self._data.info()

    @property
    def name(self) -> str:
        return self._name

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return int(self._data.shape[0])

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return int(self._data.shape[1])

    @property
    def memory(self) -> float:
        """Returns memory size of dataset in Mb"""
        return self._data.memory_usage(deep=True).sum() / (1024**2)

    @property
    def size(self) -> int:
        """Returns number cells in the dataframe."""
        return int(self.nrows * self.ncols)

    @property
    def matrix_size(self) -> int:
        """Returns number of elements in a user / item matrix"""
        return int(self.n_users * self.n_items)

    @property
    def sparsity(self) -> float:
        """Returns measure of sparsity of the data in percent"""
        return 100 - self.density

    @property
    def density(self) -> float:
        """Returns measure of density of the data in percent"""
        return self.nrows / self.matrix_size * 100

    @property
    def users(self) -> np.array:
        """Returns array of unique users"""
        return np.sort(self._data["useridx"].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data["itemidx"].unique())

    @property
    def user_item_ratio(self) -> float:
        """Returns ratio of the number of users to items."""
        return self.n_users / self.n_items

    @property
    def item_user_ratio(self) -> float:
        """Returns ratio of the number of items to users."""
        return self.n_items / self.n_users

    @property
    def n_users(self) -> int:
        """Returns number of unique users"""
        return int(self._data["userId"].nunique())

    @property
    def n_items(self) -> int:
        """Returns number of unique items."""
        return int(self._data["movieId"].nunique())

    @property
    def user_rating_counts(self) -> pd.DataFrame:
        """Returns number of ratings by user."""
        return (
            self._data["useridx"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["useridx"])
        )

    @property
    def item_rating_counts(self) -> pd.DataFrame:
        """Returns number of ratings by item."""
        return (
            self._data["itemidx"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["itemidx"])
        )

    @property
    def centered(self) -> bool:
        return self._centered

    def summarize(self) -> pd.DataFrame:
        """Returns a data frame with summary statistics"""

        d = {
            "Rows": self.nrows,
            "Columns": self.ncols,
            "Users": self.n_users,
            "Movies": self.n_items,
            "Average number of ratings per user": self.user_item_ratio,
            "Maximum Number of ratings by a user": self._data["userId"].value_counts().max(),
            "Average number of ratings per item": self.item_user_ratio,
            "Maximum Number of ratings for an item": self._data["movieId"].value_counts().max(),
            "Memory Usage (Mb)": self.memory,
            "Size": self.size,
            "Matrix Size": self.matrix_size,
            "Sparsity": self.sparsity,
            "Density": self.density,
        }

        df = pd.DataFrame.from_dict(data=d, orient="index", columns=["Count"])
        return df

    def as_dataframe(self) -> None:
        """Returns a copy of the ratings dataframe."""
        return copy(self._data)

    def as_csr(self, mean_centered: Union[str, bool] = False, axis: int = 0) -> sparse.csr_matrix:
        """Returns ratings data in csr sparse format.

        Args:
            mean_centered (str,bool): {'user', 'item', None}. Whether and how
                to mean center the ratings.
            axis (int): 0 for user per row, 1 for item per row. This creates a transposed version
                witthout the overhead of the transpose operation.
        """
        mc = self._check_mean_centered(mean_centered=mean_centered)

        data = self._extract_data(mean_centered=mc)
        sparse_params = self._get_sparse_params(data=data, axis=axis)
        csr = sparse.csr_matrix(
            (sparse_params.data, (sparse_params.row, sparse_params.col)),
            shape=(sparse_params.rows, sparse_params.cols),
        )
        return csr

    def as_csc(self, mean_centered: Union[str, bool] = False, axis: int = 0) -> sparse.csc_matrix:
        """Returns ratings data in csc sparse format

        Args:
            mean_centered (str,bool): {'user', 'item', None}. Whether and how
                to mean center the ratings.
            axis (int): 0 for user per row, 1 for item per row. This creates a transposed version
                witthout the overhead of the transpose operation.

        """

        mc = self._check_mean_centered(mean_centered=mean_centered)

        data = self._extract_data(mean_centered=mc)
        sparse_params = self._get_sparse_params(data=data, axis=axis)
        csc = sparse.csc_matrix(
            (sparse_params.data, (sparse_params.row, sparse_params.col)),
            shape=(sparse_params.rows, sparse_params.cols),
        )
        return csc

    def top_n_users(self, n: int = 10) -> pd.DataFrame:
        """Returns the users with n highest number of ratings.

        Args:
            n (int): The number of top users to return

        Returns: pd.DataFrame with userId and rating counts.

        """
        return self._data["useridx"].value_counts(sort=True).to_frame("Counts")[0:n]

    def top_n_items(self, n: int = 10) -> pd.DataFrame:
        """Returns the items with n highest number of ratings.

        Args:
            n (int): The number of top items to return

        Returns: pd.DataFrame with movieId and rating counts.

        """
        return self._data["itemidx"].value_counts(sort=True).to_frame("Counts")[0:n]

    def get_user_ratings(
        self, useridx: int, mean_centered: Union[str, bool] = False
    ) -> pd.DataFrame:
        """Returns ratings created by user.

        Args:
            useridx (int): index for the user

        Returns: pd.DataFrame
        """
        if mean_centered is not False:
            data = self._extract_data(mean_centered=mean_centered)
        else:
            data = self._data
        return data[data["useridx"] == useridx]

    def get_item_ratings(
        self, itemidx: int, mean_centered: Union[str, bool] = False
    ) -> pd.DataFrame:
        """Returns ratings for the given item

        Args:
            itemidx (int): Index for the item / movie

        Returns: pd.DataFrame
        """
        if mean_centered is not False:
            data = self._extract_data(mean_centered=mean_centered)
        else:
            data = self._data
        return data[data["itemidx"] == itemidx]

    def get_user_item_rating(
        self, useridx: int, itemidx: int, mean_centered: Union[str, bool] = False
    ) -> np.float:
        """Returns the rating given by the user useridx for item itemidx

        Args:
            useridx (int) Index for a given user
            itemidx (int): Index for a given item

        """
        if mean_centered is not False:
            data = self._extract_data(mean_centered=mean_centered)
        else:
            data = self._data
        return data[(data["useridx"] == useridx) & (data["itemidx"] == itemidx)]["rating"].values[0]

    def get_users_rated_item(self, itemidx: int) -> list:
        """Returns a list of users who have rated itemidx

        Args:
            itemidx (int): The index for the item
        """
        return self._data[self._data["itemidx"] == itemidx]["useridx"].tolist()

    def get_items_rated_user(self, useridx: int) -> list:
        """Returns a list of items rated by useridx.

        Args:
            useridx (int): The index for the user
        """
        return self._data[self._data["useridx"] == useridx]["itemidx"].tolist()

    def get_ave_user_ratings(self, useridx: int = None) -> Union[float, pd.DataFrame]:
        """Returns user average ratings

        For a user, the method returns a float. If user is not specified, a DataFrame
        of all user average ratings will be returned.

        Args:
            useridx (int): Optional. The index for the user for which average ratings are being requested.

        """
        if self._ave_user_ratings is None:
            self._ave_user_ratings = self._data.groupby("useridx")["rating"].mean().reset_index()
            self._ave_user_ratings.columns = ["useridx", "rbar"]

        if useridx is None:
            return self._ave_user_ratings
        else:
            try:
                return self._ave_user_ratings[self._ave_user_ratings["useridx"] == useridx][
                    "rbar"
                ].values[0]
            except IndexError:
                msg = f"User {useridx} has no ratings"
                self._logger.error(msg)
                raise ValueError(msg)

    def get_ave_item_ratings(self, itemidx: int = None) -> Union[float, pd.DataFrame]:
        """Returns item average ratings

        For an item, the method returns a float. If item is not specified, a DataFrame
        of all item average ratings will be returned.

        Args:
            itemidx (int): Optional. Item id for which average ratings are being requested.
        """
        if self._ave_item_ratings is None:
            self._ave_item_ratings = self._data.groupby("itemidx")["rating"].mean().reset_index()
            self._ave_item_ratings.columns = ["itemidx", "rbar"]

        if itemidx is None:
            return self._ave_item_ratings
        else:
            try:
                return self._ave_item_ratings[self._ave_item_ratings["itemidx"] == itemidx][
                    "rbar"
                ].values[0]
            except IndexError:
                msg = f"Item {itemidx} has no ratings"
                self._logger.error(msg)
                raise ValueError(msg)

    def prune(self, min_item_ratings: int = 2, min_user_ratings: int = 20) -> pd.DataFrame:
        """Returns a pruned dataset with rating counts for items and users as prescribed

        Args:
            min_item_ratings (int): Minimum number of ratings per item
            min_user_ratings (int): Minimum number of ratings per user
        """
        item_rating_counts = self.item_rating_counts
        items = item_rating_counts[item_rating_counts["n_ratings"] >= min_item_ratings][
            "itemidx"
        ].values
        user_rating_counts = self.user_rating_counts
        users = user_rating_counts[user_rating_counts["n_ratings"] >= min_user_ratings][
            "useridx"
        ].values
        self._data = self._data[
            (self._data["useridx"].isin(users)) & (self._data["itemidx"].isin(items))
        ]

    def center(self) -> pd.DataFrame:
        """Centers the ratings by average user and item ratings."""
        if self._centered is not True:
            dimensions = {
                "user": {"dim_col": "useridx", "mu_col": "mu_u", "centered_col": "rating_cu"},
                "item": {"dim_col": "itemidx", "mu_col": "mu_i", "centered_col": "rating_ci"},
            }

            for dimension in dimensions.values():
                self._center(
                    dim_col=dimension["dim_col"],
                    mu_col=dimension["mu_col"],
                    centered_col=dimension["centered_col"],
                )
            self._centered = True

    def _center(self, dim_col: str, mu_col: str, centered_col: str) -> None:
        mu = self._data.groupby(dim_col)["rating"].mean().reset_index()
        mu.columns = [dim_col, mu_col]
        self._data = self._data.merge(mu, on=dim_col, how="left")
        self._data[centered_col] = self._data["rating"] - self._data[mu_col]
        self._data = self._data.drop(columns=[mu_col])

    def _extract_data(self, mean_centered: Union[str, bool] = False) -> pd.DataFrame:
        """Extracts data raw or mean centered data, as prescribed"""
        if mean_centered and not self.centered:
            self.center()
        if mean_centered == "user":
            data = self._data[["useridx", "itemidx", "rating_cu"]]
        elif mean_centered == "item":
            data = self._data[["useridx", "itemidx", "rating_ci"]]
        else:
            data = self._data[["useridx", "itemidx", "rating"]]
        data.columns = ["useridx", "itemidx", "rating"]
        return data

    def _get_sparse_params(self, data: pd.DataFrame, axis: int = 0) -> namedtuple:
        """Returns a named tuple of row, col, data, rows, cols parameters for the sparse matrix constructor."""

        SparseParams = namedtuple("SparseParams", "row col rows cols data")

        if axis == 0:
            row = data.useridx.values
            col = data.itemidx.values
            rows = len(range(data.useridx.max() + 1))
            cols = len(range(data.itemidx.max() + 1))
        else:
            row = data.itemidx.values
            col = data.useridx.values
            rows = len(range(data.itemidx.max() + 1))
            cols = len(range(data.useridx.max() + 1))

        data = data.rating.values
        sp = SparseParams(row, col, rows, cols, data)

        return sp

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

    def _check_mean_centered(self, mean_centered: Union[str, bool] = False) -> str:
        """Checks mean_centered parameter and returns a value that can be used as a dictionary key"""
        if mean_centered is None or mean_centered is False:
            return "raw"
        elif "u" in mean_centered:
            return "user"
        elif "i" in mean_centered:
            return "item"
        else:
            msg = f"'mean_centered'={mean_centered} is invalid. Must be in [False,'u', 'user', 'i', 'item']"
            self._logger.error(msg)
            raise ValueError(msg)

    def _get_filepath(self) -> str:
        """Returns the filepath for the given mode and filename"""
        if self._mode == "test":
            return os.path.join("tests", "data", self._filename)
        else:
            return os.path.join("data", self._mode, self._filename)
