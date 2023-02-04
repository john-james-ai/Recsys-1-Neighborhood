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
# Modified   : Saturday February 4th 2023 12:08:32 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Exploratory Data Analysis Module"""
import os
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm

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
        self._user_inverted_idx = {}
        self._item_inverted_idx = {}
        self._ave_user_ratings = None
        self._ave_item_ratings = None
        self._user_rating_norms = None
        self._item_rating_norms = None
        self._centered = False

        self._preprocess()

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

    def user_inverted_idx(self, force: bool = False) -> dict:
        """Computes a user-item inverted index

        Args:
            force (bool): Whether to compute if the index has already been computed.
        """
        if force or not self._user_inverted_idx:
            for user in self.users:
                self._user_inverted_idx[user] = self.get_items(user)
        return self._user_inverted_idx

    def item_inverted_idx(self, force: bool = False) -> dict:
        """Computes a user-item inverted index

        Args:
            force (bool): Whether to compute if the index has already been computed.
        """
        if force or not self._item_inverted_idx:
            for item in self.items:
                self._item_inverted_idx[item] = self.get_users(item)
        return self._item_inverted_idx

    def get_users_items_ratings(
        self, items: list, users: list, centered_by: str = None
    ) -> pd.DataFrame:
        """Subsets ratings by users and items.

        The original ratings are returned, unless normalized by is set to 'user', or 'item'; whereby,
        the ratings are normalized by the user average rating or the item average rating, respectively.

        Args:
            items (List[int]): List of ids for items of interest
            users (List[int]): List of ids for users of interest
            centered_by (str): Optional. One of 'item', 'user' or None. Default is None.

        """
        data = self._get_data(centered_by=centered_by)
        return data[(data["userId"].isin(users)) & (self._data["movieId"].isin(items))]

    def get_user_ratings(self, user: int, centered_by: str = None) -> pd.DataFrame:
        """Gets all user ratings.

        Args:
            user (int): The id for the user for whom the ratings are being returned.
        """
        data = self._get_data(centered_by=centered_by)

        return data[data["userId"] == user].sort_values(by="movieId", ascending=True, axis=0)

    def get_item_ratings(self, item: int, centered_by: str = None) -> pd.DataFrame:
        """Gets all item ratings.

        Args:
            item (int): The id for item for which the ratings are being returned.
            centered_by (str): Optional. One of 'item', 'user' or None. Default is None.

        """
        data = self._get_data(centered_by=centered_by)

        return data[data["movieId"] == item].sort_values(by="userId", ascending=True, axis=0)

    def get_ave_user_ratings(self, user: int = None) -> Union[float, pd.DataFrame]:
        """Returns user average ratings

        For a user, the method returns a float. If user is not specified, a DataFrame
        of all user average ratings will be returned.

        Args:
            user (int): Optional. User for which average ratings are being requested.
        """
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
        if item is None:
            return self._ave_item_ratings
        else:
            return self._ave_item_ratings[self._ave_item_ratings["movieId"] == item]["rbar"].values[
                0
            ]

    def get_user_rating_norms(self, user: int = None, centered_by: str = None) -> pd.DataFrame:
        rating_map = {None: "rating_l2", "user": "rating_cbu_l2", "item": "rating_cbi_l2"}

        if self._user_rating_norms is None:
            self._compute_user_rating_norms()

        rating_type = rating_map.get(centered_by, None)
        if rating_type is not None:
            norm = self._user_rating_norms[["userId", rating_type]]
            norm.columns = ["userId", "l2"]
            if user is None:
                return norm
            else:
                return norm[norm["userId"] == user]["l2"].values
        else:
            msg = "Invalid value for 'centered_by'. Must be in ['user', 'item', None]."
            self._logger.error(msg)
            raise ValueError(msg)

    def get_item_rating_norms(self, item: int = None, centered_by: str = None) -> pd.DataFrame:
        rating_map = {None: "rating_l2", "user": "rating_cbu_l2", "item": "rating_cbi_l2"}

        if self._item_rating_norms is None:
            self._compute_item_rating_norms()

        rating_type = rating_map.get(centered_by, None)
        if rating_type is not None:
            norm = self._item_rating_norms[["movieId", rating_type]]
            norm.columns = ["movieId", "l2"]
            if item is None:
                return norm
            else:
                return norm[norm["movieId"] == item]["l2"].values
        else:
            msg = "Invalid value for 'centered_by'. Must be in ['user', 'item', None]."
            self._logger.error(msg)
            raise ValueError(msg)

    def _compute_user_rating_norms(self) -> pd.DataFrame:
        """Returns l2 norms of raw, user centered, and item centered ratings for each user"""
        tqdm.pandas()
        self.center_ratings()
        data = self._data[["userId", "rating", "rating_cbu", "rating_cbi"]]
        self._user_rating_norms = (
            data.groupby("userId")
            .progress_apply(
                lambda x: np.sqrt(
                    np.sum(np.square(x[["rating", "rating_cbu", "rating_cbi"]]), axis=0)
                )
            )
            .reset_index()
        )
        self._user_rating_norms.columns = ["userId", "rating_l2", "rating_cbu_l2", "rating_cbi_l2"]

    def _compute_item_rating_norms(self) -> pd.DataFrame:
        """Returns l2 norms of raw, user centered, and item centered ratings for each item"""
        tqdm.pandas()
        self.center_ratings()
        data = self._data[["movieId", "rating", "rating_cbu", "rating_cbi"]]
        self._item_rating_norms = (
            data.groupby("movieId")
            .progress_apply(
                lambda x: np.sqrt(
                    np.sum(np.square(x[["rating", "rating_cbu", "rating_cbi"]]), axis=0)
                )
            )
            .reset_index()
        )
        self._item_rating_norms.columns = ["movieId", "rating_l2", "rating_cbu_l2", "rating_cbi_l2"]

    def center_ratings(self) -> None:
        """Adds rating centered by average user rating, and average item rating if not already done."""
        if self._centered is False:
            self._center_by_ave_user_rating()
            self._center_by_ave_item_rating()
            self._centered = True

    def _preprocess(self) -> None:
        self._index()
        self._categorize()
        self._compute_ave_item_ratings()
        self._compute_ave_user_ratings()

    def _index(self) -> None:
        """Map user and item indexes to ids."""
        # Create User Map
        userId = np.sort(self._data["userId"].unique())
        useridx = range(len(userId))
        u = {"userId": userId, "useridx": useridx}
        u = pd.DataFrame(data=u)

        # Create Item Map
        movieId = np.sort(self._data["movieId"].unique())
        itemidx = range(len(movieId))
        i = {"movieId": movieId, "itemidx": itemidx}
        i = pd.DataFrame(data=i)

        # Install New Indices
        self._data = self._data.merge(u, on="userId", how="left")
        self._data = self._data.merge(i, on="movieId", how="left")

    def _categorize(self) -> None:
        self._data = self._data.astype(
            {
                "userId": "category",
                "movieId": "category",
                "useridx": "category",
                "itemidx": "category",
            }
        )

    def _compute_ave_user_ratings(self) -> None:
        """Computes average user ratings."""
        self._ave_user_ratings = (
            self._data.groupby("userId")["rating"].mean().to_frame().reset_index()
        )
        self._ave_user_ratings.columns = ["userId", "rbar"]

    def _compute_ave_item_ratings(self) -> None:
        """Computes average item ratings."""
        self._ave_item_ratings = (
            self._data.groupby("movieId")["rating"].mean().to_frame().reset_index()
        )
        self._ave_item_ratings.columns = ["movieId", "rbar"]

    def _center_by_ave_user_rating(self) -> None:
        """Adds a column of ratings values normalized by average user rating."""
        self._data = self._data.merge(self._ave_user_ratings, on="userId", how="left")
        self._data["rating_cbu"] = self._data["rating"].values - self._data["rbar"].values
        self._data = self._data.drop(columns=["rbar"])

    def _center_by_ave_item_rating(self) -> None:
        """Creates a dataframe containing each item and its average rating."""
        self._data = self._data.merge(self._ave_item_ratings, on="movieId", how="left")
        self._data["rating_cbi"] = self._data["rating"].values - self._data["rbar"].values
        self._data = self._data.drop(columns=["rbar"])

    def _get_data(self, centered_by: str = None) -> pd.DataFrame:
        """Returns ratings data with requested normalization.

        centered_by (str): Either 'item', 'user', or None.
        """
        if centered_by is not None:
            self.center_ratings()  # Only happens once.
            if centered_by == "user":
                data = self._data[["userId", "movieId", "rating_cbu"]].copy()
            elif centered_by == "item":
                data = self._data[["userId", "movieId", "rating_cbi"]].copy()
            else:
                msg = f"Value for 'centered_by' = {centered_by} is invalid. Must be 'item', 'user', or None."
                self._logger.error(msg)
                raise ValueError(msg)
        else:
            data = self._data[["userId", "movieId", "rating"]].copy()
        data.columns = ["userId", "movieId", "rating"]
        return data
