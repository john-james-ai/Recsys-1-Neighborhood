#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataset/movielens.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 06:05:25 pm                                                  #
# Modified   : Sunday March 19th 2023 04:13:29 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
import warnings
from copy import deepcopy

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import numpy as np
import pandas as pd

from recsys.dataset.base import Dataset

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------ #
class MovieLens(Dataset):
    """Object containing interaction data.

    Args:
        name (str): Name of the matrix in lowercase
        desc (str): desc of the matrix
        filepath (str): The persistence location for this object.
        data (str): Pandas DataFrame containing rating interaction data.
    """

    __ITEMID = "movieId"
    __USERID = "userId"
    __RATING = "rating"
    __RATING_USER_CENTERED = "rating_cu"
    __RATING_ITEM_CENTERED = "rating_ci"
    __TIMESTAMP = "timestamp"

    def __init__(
        self,
        name: str,
        desc: str,
        filepath: str,
        data: pd.DataFrame,
    ) -> None:
        super().__init__(name=name, desc=desc, filepath=filepath, data=data)

        self._profiled = False
        self._summary = None
        self._nrows = None
        self._ncols = None
        self._size = None
        self._n_users = None
        self._n_items = None
        self._interaction_matrix_size = None
        self._sparsity = None
        self._density = None
        self._memory = None

        self._summarize()

    @property
    def shape(self) -> tuple:
        return ()

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
    def columns(self) -> np.array:
        return self._data.columns

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def size(self) -> int:
        return self._size

    @property
    def interaction_matrix_size(self) -> int:
        return self._interaction_matrix_size

    @property
    def users(self) -> np.array:
        """Returns array of unique users"""
        return np.sort(self._data[MovieLens.__USERID].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data[MovieLens.__ITEMID].unique())

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
            self._data[MovieLens.__USERID]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=[MovieLens.__USERID])
        )

    @property
    def user_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of user rating frequency"""
        return self.user_rating_frequency["n_ratings"].describe().to_frame().T

    @property
    def item_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by item."""
        return (
            self._data[MovieLens.__ITEMID]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=[MovieLens.__ITEMID])
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

        return self._data[self._data[MovieLens.__USERID] == useridx]

    def get_item_ratings(self, itemidx: int) -> pd.DataFrame:
        """Returns ratings for the given item
        Args:
            itemidx (int): Index for the item / movie
        Returns: pd.DataFrame
        """
        return self._data[self._data[MovieLens.__ITEMID] == itemidx]

    def get_users_rated_item(self, itemidx: int) -> list:
        """Returns a list of users who have rated itemidx
        Args:
            itemidx (int): The index for the item
        """
        return self._data[self._data[MovieLens.__ITEMID] == itemidx][MovieLens.__USERID].tolist()

    def get_items_rated_user(self, useridx: int) -> list:
        """Returns a list of items rated by useridx.
        Args:
            useridx (int): The index for the user
        """
        return self._data[self._data[MovieLens.__USERID] == useridx][MovieLens.__ITEMID].tolist()

    def get_items_rated_users(self, u: int, v: int) -> set:
        """Returns a list of items rated by both u and v.

        Args:
            u (int): A user index
            v (int): A user index
        """
        Iu = self.get_items_rated_user(useridx=u)
        Iv = self.get_items_rated_user(useridx=v)
        return set(Iu).intersection(Iv)

    def get_users_rated_items(self, i: int, j: int) -> set:
        """Returns a list of users who have rated both items i and j.

        Args:
            i (int): An item index
            j (int): An item index
        """
        Ui = self.get_users_rated_item(itemidx=i)
        Uj = self.get_users_rated_item(itemidx=j)
        return set(Ui).intersection(Uj)

    def compare(self, other: MovieLens) -> pd.DataFrame:
        """Compare this and another MovieLens returning descriptive statistics.

        Args:
            other (MovieLens): The other interaction matrix which to compare.
        """
        df1 = self._summary
        df2 = other.summary()
        both = pd.concat([df1, df2], axis=1)
        both["% change"] = (df1[self._name] - df2[other.name]) / df1[self._name] * 100
        return both

    def to_df(self) -> pd.DataFrame:
        """Returns the nonzero values in dataframe format"""
        return deepcopy(self._data)

    def to_csr(self, centered_by: str = None) -> csr_matrix:
        """Produces a csr matrix

        Args:
            centered_by (str): Valid values in [None, 'user', 'item']. Default is None

        Returns: scipy.sparse.csr_matrix

        """

        if centered_by is None:
            col = MovieLens.__RATING
        elif "u" in centered_by.lower():
            col = MovieLens.__RATING_USER_CENTERED
        else:
            col = MovieLens.__RATING_ITEM_CENTERED

        rows = self._data[MovieLens.__USERID]
        cols = self._data[MovieLens.__ITEMID]
        data = self._data[col]
        return csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_csc(self, centered_by: str = None) -> csc_matrix:
        """Produces a csr matrix

        Args:
            centered_by (str): Valid values in [None, 'user', 'item']. Default is None

        Returns: scipy.sparse.csc_matrix

        """

        if centered_by is None:
            col = MovieLens.__RATING
        elif "user" in centered_by:
            col = MovieLens.__RATING_USER_CENTERED
        else:
            col = MovieLens.__RATING_ITEM_CENTERED

        rows = self._data[MovieLens.__USERID]
        cols = self._data[MovieLens.__ITEMID]
        data = self._data[col]
        return csc_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_coo(self, centered_by: str = None) -> coo_matrix:
        """Produces a csr matrix

        Args:
            centered_by (str): Valid values in [None, 'user', 'item']. Default is None

        Returns: scipy.sparse.csc_matrix

        """
        if centered_by is None:
            col = MovieLens.__RATING
        elif "user" in centered_by:
            col = MovieLens.__RATING_USER_CENTERED
        else:
            col = MovieLens.__RATING_ITEM_CENTERED

        rows = self._data[MovieLens.__USERID]
        cols = self._data[MovieLens.__ITEMID]
        data = self._data[col]
        return coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_binary(self) -> csr_matrix:
        """Returns a user/item interaction matrix in csr format"""
        df = self._data[[MovieLens.__USERID, MovieLens.__ITEMID]]
        df["interaction"] = 1
        rows = df[MovieLens.__USERID]
        cols = df[MovieLens.__ITEMID]
        data = df["interaction"]
        return csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def _summarize(self) -> None:
        """Runs a data profile including basic summary statistics"""
        if not self._profiled:
            self._logger.debug("Computing descriptive statistics....")
            # Computes basic statistics
            self._nrows = self._data.shape[0]
            self._ncols = self._data.shape[1]
            self._size = self._nrows * self._ncols
            self._memory = round(self._data.memory_usage(deep=True).sum() / 1024**2, 3)
            self._n_users = int(self._data[MovieLens.__USERID].nunique())
            self._n_items = int(self._data[MovieLens.__ITEMID].nunique())
            self._user_item_ratio = self.user_item_ratio
            self._item_user_ratio = self.item_user_ratio
            self._mean_ratings_per_user = self._nrows / self._n_users
            self._mean_ratings_per_item = self._nrows / self._n_items
            self._interaction_matrix_size = int(self._n_users * self._n_items)
            self._density = self._nrows / (self._n_users * self._n_items) * 100
            self._sparsity = 100 - self._density
            self._memory = self._data.memory_usage(deep=True).sum()
            self._max_ratings_per_user = self.user_rating_frequency["n_ratings"].max()
            self._max_ratings_per_item = self.item_rating_frequency["n_ratings"].max()
            self._min_ratings_per_user = self.user_rating_frequency["n_ratings"].min()
            self._min_ratings_per_item = self.item_rating_frequency["n_ratings"].min()

            d = {}
            # d["name"] = self._name
            # d["type"] = self.__class__.__name__
            # d["desc"] = self._desc
            d["nrows"] = self._data.shape[0]
            d["ncols"] = self._data.shape[1]
            d["n_users"] = self._n_users
            d["n_items"] = self._n_items
            d["max_ratings_per_user"] = self._max_ratings_per_user
            d["mean_ratings_per_user"] = self._mean_ratings_per_user
            d["min_ratings_per_user"] = self._min_ratings_per_user
            d["max_ratings_per_item"] = self._max_ratings_per_item
            d["mean_ratings_per_item"] = self._mean_ratings_per_item
            d["min_ratings_per_item"] = self._min_ratings_per_item
            d["user_item_ratio"] = self._user_item_ratio
            d["item_user_ratio"] = self._item_user_ratio
            d["size"] = self._size
            d["interaction_matrix_size"] = self._interaction_matrix_size
            d["memory"] = self._memory
            d["sparsity"] = self._sparsity
            d["density"] = self._density

            self._summary = pd.DataFrame.from_dict(data=d, orient="index", columns=[self._name])

        self._profiled = True
