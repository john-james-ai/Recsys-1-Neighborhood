#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/rating.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 12:40:31 am                                               #
# Modified   : Sunday March 5th 2023 10:19:51 pm                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
import numpy as np
import pandas as pd

from recsys.dataset.base import Dataset


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
        self._profiled = False
        self._summary = None
        self._nrows = None
        self._ncols = None
        self._size = None
        self._n_users = None
        self._n_items = None
        self._utility_matrix_size = None
        self._sparsity = None
        self._density = None
        self._memory = None

        self.reindex()
        self.center()
        self._rearrange()
        self._summarize()

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
    def user_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by user."""
        return (
            self._data["useridx"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["useridx"])
        )

    @property
    def user_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of user rating frequency"""
        return self.user_rating_frequency["n_ratings"].describe().to_frame().T

    @property
    def item_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by item."""
        return (
            self._data["itemidx"]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=["itemidx"])
        )

    @property
    def item_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of item rating frequency"""
        return self.item_rating_frequency["n_ratings"].describe().to_frame().T

    def get_user_ratings(self, useridx: int) -> pd.DataFrame:
        """Returns ratings created by user.
        Args:
            useridx (int): index for the user
        Returns: pd.DataFrame
        """

        return self._data[self._data["useridx"] == useridx]

    def get_item_ratings(self, itemidx: int) -> pd.DataFrame:
        """Returns ratings for the given item
        Args:
            itemidx (int): Index for the item / movie
        Returns: pd.DataFrame
        """
        return self._data[self._data["itemidx"] == itemidx]

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

    def center(
        self, user_ctr_col: str = "rating_ctr_user", item_ctr_col: str = "rating_ctr_item"
    ) -> None:
        """Centers rating by user and item average rating."""
        self._center(by="useridx", col=user_ctr_col)
        self._center(by="itemidx", col=item_ctr_col)

    def _center(self, by: str, col: str) -> None:
        """Centers ratings by value indicated"""

        # Obtain average user ratings
        rbar = self._data.groupby(by=by)["rating"].mean().reset_index()
        rbar.columns = [by, "rbar"]

        # Merge with ratings dataset
        self._data = self._data.merge(rbar, on=by, how="left")

        # Compute centered rating and drop the average rating column
        self._data[col] = self._data["rating"] - self._data["rbar"]
        self._data = self._data.drop(columns=["rbar"])

    def _summarize(self) -> None:
        """Runs a data profile including basic summary statistics"""
        if not self._profiled:
            # Computes basic statistics
            self._nrows = self._data.shape[0]
            self._ncols = self._data.shape[1]
            self._size = self._nrows * self._ncols
            self._memory = round(self._data.memory_usage(deep=True).sum() / 1024**2, 3)
            self._n_users = int(self._data["useridx"].nunique())
            self._n_items = int(self._data["itemidx"].nunique())
            self._user_item_ratio = self.user_item_ratio
            self._item_user_ratio = self.item_user_ratio
            self._mean_ratings_per_user = self._nrows / self._n_users
            self._mean_ratings_per_item = self._nrows / self._n_items
            self._utility_matrix_size = int(self._n_users * self._n_items)
            self._sparsity = self._nrows / self._size * 100
            self._density = 100 - self._sparsity
            self._memory = self._data.memory_usage(deep=True).sum()
            self._max_ratings_per_user = self.user_rating_frequency["n_ratings"].max()
            self._max_ratings_per_item = self.item_rating_frequency["n_ratings"].max()

            d = {}
            # d["name"] = self._name
            # d["type"] = self.__class__.__name__
            # d["description"] = self._description
            d["nrows"] = self._data.shape[0]
            d["ncols"] = self._data.shape[1]
            d["n_users"] = self._n_users
            d["n_items"] = self._n_items
            d["max_ratings_per_user"] = self._max_ratings_per_user
            d["mean_ratings_per_user"] = self._mean_ratings_per_user
            d["max_ratings_per_item"] = self._max_ratings_per_item
            d["mean_ratings_per_item"] = self._mean_ratings_per_item
            d["user_item_ratio"] = self._user_item_ratio
            d["item_user_ratio"] = self._item_user_ratio
            d["size"] = self._size
            d["utility_matrix_size"] = self._utility_matrix_size
            d["memory"] = self._memory
            d["sparsity"] = self._sparsity
            d["density"] = self._density

            self._summary = pd.DataFrame.from_dict(data=d, orient="index", columns=[self._name])

        self._profiled = True

    def reindex(self) -> None:
        if "itemidx" in self._data.columns:
            msg = "Warning. The dataset has already been reindexed."
            self._logger.error(msg)
        else:
            self._reindex(id="userId", to="useridx")
            self._reindex(id="movieId", to="itemidx")

    def _reindex(self, id: str = "movieId", to: str = "itemidx") -> None:
        """Creates sequential ids for users and movies."""
        # Get unique user or movie ids.
        features = np.sort(self._data[id].unique())
        features = pd.DataFrame(data=features, columns=[id])
        features.reset_index(inplace=True)
        features = features.rename(columns={"index": to})
        self._data = self._data.merge(features, how="left", on=id)

    def _rearrange(self) -> None:
        cols = [
            "userId",
            "useridx",
            "movieId",
            "itemidx",
            "rating",
            "rating_ctr_user",
            "rating_ctr_item",
            "timestamp",
        ]
        self._data = self._data[cols]

    def compare(self, other: RatingsDataset) -> pd.DataFrame:
        df1 = self._summary
        df2 = other.summary()
        both = pd.concat([df1, df2], axis=1)
        both["% change"] = (df1[self._name] - df2[other.name]) / df1[self._name] * 100
        return both
