#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/feature/interaction.py                                                      #
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

from recsys.matrix.base import Matrix

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------ #
class InteractionMatrix(Matrix):
    """Object containing interaction data.

    Args:
        name (str): Name of the matrix in lowercase
        desc (str): desc of the matrix
        filepath (str): The persistence location for this object.
        data (str): Pandas DataFrame containing rating interaction data.
    """

    __ITEMIDX = "itemidx"
    __USERIDX = "useridx"
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
        super().__init__()
        self._name = name
        self._desc = desc
        self._filepath = filepath
        self._data = data

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
        self.normalize()
        self._summarize()

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        return self._desc

    @property
    def filepath(self) -> str:
        return self._filepath

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
        return np.sort(self._data[InteractionMatrix.__USERIDX].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data[InteractionMatrix.__ITEMIDX].unique())

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
            self._data[InteractionMatrix.__USERIDX]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=[InteractionMatrix.__USERIDX])
        )

    @property
    def user_rating_frequency_distribution(self) -> pd.DataFrame:
        """Distribution of user rating frequency"""
        return self.user_rating_frequency["n_ratings"].describe().to_frame().T

    @property
    def item_rating_frequency(self) -> pd.DataFrame:
        """Returns number of ratings by item."""
        return (
            self._data[InteractionMatrix.__ITEMIDX]
            .value_counts()
            .to_frame(name="n_ratings")
            .reset_index(names=[InteractionMatrix.__ITEMIDX])
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

        return self._data[self._data[InteractionMatrix.__USERIDX] == useridx]

    def get_item_ratings(self, itemidx: int) -> pd.DataFrame:
        """Returns ratings for the given item
        Args:
            itemidx (int): Index for the item / movie
        Returns: pd.DataFrame
        """
        return self._data[self._data[InteractionMatrix.__ITEMIDX] == itemidx]

    def get_users_rated_item(self, itemidx: int) -> list:
        """Returns a list of users who have rated itemidx
        Args:
            itemidx (int): The index for the item
        """
        return self._data[self._data[InteractionMatrix.__ITEMIDX] == itemidx][
            InteractionMatrix.__USERIDX
        ].tolist()

    def get_items_rated_user(self, useridx: int) -> list:
        """Returns a list of items rated by useridx.
        Args:
            useridx (int): The index for the user
        """
        return self._data[self._data[InteractionMatrix.__USERIDX] == useridx][
            InteractionMatrix.__ITEMIDX
        ].tolist()

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

    def normalize(self) -> None:
        """Normalizes ratings by centering on average item and user rating."""
        self._center(by="user")
        self._center(by="item")
        self._arrange_cols()

    def compare(self, other: InteractionMatrix) -> pd.DataFrame:
        """Compare this and another InteractionMatrix returning descriptive statistics.

        Args:
            other (InteractionMatrix): The other interaction matrix which to compare.
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
            col = InteractionMatrix.__RATING
        elif "u" in centered_by.lower():
            col = InteractionMatrix.__RATING_USER_CENTERED
        else:
            col = InteractionMatrix.__RATING_ITEM_CENTERED

        rows = self._data[InteractionMatrix.__USERIDX]
        cols = self._data[InteractionMatrix.__ITEMIDX]
        data = self._data[col]
        return csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_csc(self, centered_by: str = None) -> csc_matrix:
        """Produces a csr matrix

        Args:
            centered_by (str): Valid values in [None, 'user', 'item']. Default is None

        Returns: scipy.sparse.csc_matrix

        """

        if centered_by is None:
            col = InteractionMatrix.__RATING
        elif "user" in centered_by:
            col = InteractionMatrix.__RATING_USER_CENTERED
        else:
            col = InteractionMatrix.__RATING_ITEM_CENTERED

        rows = self._data[InteractionMatrix.__USERIDX]
        cols = self._data[InteractionMatrix.__ITEMIDX]
        data = self._data[col]
        return csc_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_coo(self, centered_by: str = None) -> coo_matrix:
        """Produces a csr matrix

        Args:
            centered_by (str): Valid values in [None, 'user', 'item']. Default is None

        Returns: scipy.sparse.csc_matrix

        """
        if centered_by is None:
            col = InteractionMatrix.__RATING
        elif "user" in centered_by:
            col = InteractionMatrix.__RATING_USER_CENTERED
        else:
            col = InteractionMatrix.__RATING_ITEM_CENTERED

        rows = self._data[InteractionMatrix.__USERIDX]
        cols = self._data[InteractionMatrix.__ITEMIDX]
        data = self._data[col]
        return coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def to_binary(self) -> csr_matrix:
        """Returns a user/item interaction matrix in csr format"""
        df = self._data[[InteractionMatrix.__USERID, InteractionMatrix.__ITEMID]]
        df["interaction"] = 1
        rows = df[InteractionMatrix.__USERIDX]
        cols = df[InteractionMatrix.__ITEMIDX]
        data = df["interaction"]
        return csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def reindex(self) -> None:
        self._logger.debug("Reindexing...")
        if InteractionMatrix.__ITEMIDX in self._data.columns:
            msg = "The dataset has already been reindexed."
            self._logger.info(msg)
        else:
            self._reindex(id=InteractionMatrix.__USERID, to=InteractionMatrix.__USERIDX)
            self._reindex(id=InteractionMatrix.__ITEMID, to=InteractionMatrix.__ITEMIDX)

    def _center(self, by: str = "user", epsilon: float = 1e-9) -> None:
        """Centers users by the user or item average rating.

        Note: All centered ratings will have an adjustment added to reduce
        probability of a zero rating, as zeros, explicit or not, are treated
        as zeros in sparse matrices.

        The columns for the centered ratings are:
            user: 'rating_cu'
            item: 'rating_ci'

        Args:
            by (str): Valid values in ['user', 'item']
            epsilon (float): An adjustment to avoid zero mean rating.

        """
        if "user" in by:
            by = InteractionMatrix.__USERIDX
            col = InteractionMatrix.__RATING_USER_CENTERED
        else:
            by = InteractionMatrix.__ITEMIDX
            col = InteractionMatrix.__RATING_ITEM_CENTERED

        if col in self._data.columns:
            msg = f"Ratings have already been centered {by} in {col}."
            self._logger.info(msg)
        else:
            self._logger.debug(f"Centering ratings by {by} and storing in {col}.")

            # Obtain average ratings
            rbar = self._data.groupby(by=by)[InteractionMatrix.__RATING].mean().reset_index()
            rbar.columns = [by, "rbar"]

            # Merge with ratings dataset
            self._data = self._data.merge(rbar, on=by, how="left")

            # Compute centered rating and drop the average rating column
            self._data[col] = self._data[InteractionMatrix.__RATING] - self._data["rbar"] + epsilon
            self._data = self._data.drop(columns=["rbar"])

    def _summarize(self) -> None:
        """Runs a data profile including basic summary statistics"""
        if not self._profiled:
            self._logger.debug("Computing descriptive statistics....")
            # Computes basic statistics
            self._nrows = self._data.shape[0]
            self._ncols = self._data.shape[1]
            self._size = self._nrows * self._ncols
            self._memory = round(self._data.memory_usage(deep=True).sum() / 1024**2, 3)
            self._n_users = int(self._data[InteractionMatrix.__USERIDX].nunique())
            self._n_items = int(self._data[InteractionMatrix.__ITEMIDX].nunique())
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
            # d["desc"] = self._desc
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

    def _reindex(self, id: str, to: str) -> None:
        """Creates sequential ids for users and movies."""
        # Get unique user or movie ids.
        features = np.sort(self._data[id].unique())
        features = pd.DataFrame(data=features, columns=[id])
        features.reset_index(inplace=True)
        features = features.rename(columns={"index": to})
        self._data = self._data.merge(features, how="left", on=id)

    def _arrange_cols(self) -> None:
        cols = [
            InteractionMatrix.__USERIDX,
            InteractionMatrix.__USERID,
            InteractionMatrix.__ITEMIDX,
            InteractionMatrix.__ITEMID,
            InteractionMatrix.__RATING,
            InteractionMatrix.__RATING_USER_CENTERED,
            InteractionMatrix.__RATING_ITEM_CENTERED,
            InteractionMatrix.__TIMESTAMP,
        ]
        self._data = self._data[cols]
