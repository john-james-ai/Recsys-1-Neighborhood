#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/domain/rating.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 07:10:48 pm                                             #
# Modified   : Sunday February 19th 2023 05:01:15 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np
import pandas as pd
from typing import Union

from recsys.domain.base import Dataset
from recsys.dal.dto import DatasetDTO


# ------------------------------------------------------------------------------------------------ #
class RatingsDataset(Dataset):
    """Dataset containing movielens ratings data"""

    def __init__(
        self,
        data: Union[pd.DataFrame, dict],
        name: str,
        description: str,
        stage: str,
        env: str = None,
    ) -> None:
        super().__init__(data=data, name=name, description=description, stage=stage, env=env)

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return int(self._data.shape[0])

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return int(self._data.shape[1])

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

    def as_dto(self) -> DatasetDTO:
        """Returns a Data Transfer Object representation of the entity."""
        return DatasetDTO(
            type=self._type,
            name=self._name,
            description=self._description,
            stage=self._stage,
            env=self._env,
            memory=self._memory,
            cost=self._cost,
        )
