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
# Modified   : Thursday February 2nd 2023 09:08:46 pm                                              #
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
        self._name = os.path.basename(filepath)
        self._filepath = filepath
        self._data = IOService.read(filepath)
        self._user_inverted_idx = {}
        self._item_inverted_idx = {}
        self.preprocess()

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
    def size(self) -> float:
        """Returns memory size of dataset in Mb"""
        return self._data.memory_usage(deep=True).sum() / (1024**2)

    @property
    def utility_matrix_size(self) -> int:
        return self.nrows * self.ncols

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
            "Size": self._data.memory_usage(deep=True).sum(),
            "Rows": self._data.shape[0],
            "Columns": self._data.shape[1],
            "Users": self._data["userId"].nunique(),
            "Movies": self._data["movieId"].nunique(),
            "Utility Matrix Size": self.utility_matrix_size,
            "Utility Matrix Memory Size (Mb)": self.utility_matrix_size / 1024**2,
            "Utility Matrix Sparsity": self._data.shape[0] / self.utility_matrix_size,
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
        self, items: list, users: list, normalized_by: str = None
    ) -> pd.DataFrame:
        """Subsets ratings by users and items.

        The original ratings are returned, unless normalized by is set to 'user', or 'item'; whereby,
        the ratings are normalized by the user average rating or the item average rating, respectively.

        Args:
            items (List[int]): List of ids for items of interest
            users (List[int]): List of ids for users of interest
            normalized_by (str): Optional. One of 'item', 'user' or None. Default is None.

        """
        data = self._get_data(normalized_by=normalized_by)
        return data[(data["userId"].isin(users)) & (self._data["movieId"].isin(items))]

    def get_user_ratings(self, user: int, normalized_by: str = None) -> pd.DataFrame:
        """Gets all user ratings.

        Args:
            user (int): The id for the user for whom the ratings are being returned.
        """
        data = self._get_data(normalized_by=normalized_by)

        return data[data["userId"] == user].sort_values(by="movieId", ascending=True, axis=0)

    def get_item_ratings(self, item: int, normalized_by: str = None) -> pd.DataFrame:
        """Gets all item ratings.

        Args:
            item (int): The id for item for which the ratings are being returned.
            normalized_by (str): Optional. One of 'item', 'user' or None. Default is None.

        """
        data = self._get_data(normalized_by=normalized_by)

        return data[data["movieId"] == item].sort_values(by="userId", ascending=True, axis=0)

    def get_user_ratings_norm(self, user: int = None, normalized_by: str = None) -> pd.DataFrame:
        """Returns the L2 norm of user ratings for the designated user or all users if user is None

        For a single user, a float is returned. If user is None, a DataFrame containing
        user rating norms is returned.

        Args:
            user (int): Id for the user for whom the L2 norm of ratings is being requested. Optional
            normalized_by (str): Optional. One of 'item', 'user' or None. Default is None.
        """
        data = self._get_data(normalized_by=normalized_by)
        norms = (
            data.groupby("userId")
            .apply(lambda x: np.sqrt(np.sum(np.square(x["rating"].values))))
            .reset_index()
        )
        if user is None:
            return norms
        else:
            return norms[norms["userId"] == user].values[0]

    def get_item_ratings_norm(self, item: int = None, normalized_by: str = None) -> pd.DataFrame:
        """Returns the L2 norm of item ratings for the designated item or all items if item is None

        For a single item, a float is returned. If item is None, a DataFrame containing
        item rating norms is returned.

        Args:
            item (int): Id for the item for which the L2 norm of ratings is being requested. Optional
            normalized_by (str): Optional. One of 'item', 'user' or None. Default is None.
        """
        data = self._get_data(normalized_by=normalized_by)
        norms = (
            data.groupby("movieId")
            .apply(lambda x: np.sqrt(np.sum(np.square(x["rating"].values))))
            .reset_index()
        )
        if item is None:
            return norms
        else:
            return norms[norms["movieId"] == item].values[0]

    def preprocess(self) -> None:
        self._normalize_by_user_ratings()

        self._normalize_by_item_ratings()

    def _normalize_by_user_ratings(self) -> None:
        """Adds a column of ratings values normalized by average user rating."""
        rbar = self._data.groupby("userId")["rating"].mean().reset_index()
        rbar.columns = ["userId", "rbar"]
        self._data = self._data.merge(rbar, on="userId", how="left")
        self._data["rating_nbu"] = self._data["rating"] - self._data["rbar"]
        self._data = self._data.drop(columns=["rbar"])

    def _normalize_by_item_ratings(self) -> None:
        """Creates a dataframe containing each item and its average rating."""
        rbar = self._data.groupby("movieId")["rating"].mean().reset_index()
        rbar.columns = ["movieId", "rbar"]
        self._data = self._data.merge(rbar, on="movieId", how="left")
        self._data["rating_nbi"] = self._data["rating"] - self._data["rbar"]
        self._data = self._data.drop(columns=["rbar"])

    def _get_data(self, normalized_by: str = None) -> pd.DataFrame:
        """Returns ratings data with requested normalization.

        normalized_by (str): Either 'item', 'user', or None.
        """
        if normalized_by is not None:
            if normalized_by == "user":
                data = self._data[["userId", "movieId", "rating_nbu"]]
            elif normalized_by == "item":
                data = self._data[["userId", "movieId", "rating_nbi"]]
            else:
                msg = f"Value for 'normalized_by' = {normalized_by} is invalid. Must be 'item', 'user', or None."
                self._logger.error(msg)
                raise ValueError(msg)
        else:
            data = self._data[["userId", "movieId", "rating"]]
        data.columns = ["userId", "movieId", "rating"]
        return data
