#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/dataset.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 07:10:48 pm                                             #
# Modified   : Wednesday February 22nd 2023 01:25:45 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np
import pandas as pd

from recsys.data.base import Data
from recsys.dal.dto import DatasetDTO


# ------------------------------------------------------------------------------------------------ #
class Dataset(Data):
    """Dataset containing movielens ratings data"""

    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        description: str,
        stage: str,
    ) -> None:
        super().__init__(name=name, description=description, stage=stage)
        self._data = data
        self._summary = None
        self._summarize()

    @property
    def users(self) -> np.array:
        """Returns array of unique users"""
        return np.sort(self._data["userId"].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data["movieId"].unique())

    def get_user_ratings(self, userId: int) -> pd.DataFrame:
        """Returns ratings created by user.
        Args:
            userId (int): index for the user
        """
        return self._data[self._data["userId"] == userId]

    def get_item_ratings(self, movieId: int) -> pd.DataFrame:
        """Returns ratings for the given item
        Args:
            movieId (int): Index for the item / movie
        Returns: pd.DataFrame
        """
        return self._data[self._data["movieId"] == movieId]

    def list_raters_for_item(self, movieId: int) -> list:
        """Returns a list of users who have rated movieId
        Args:
            movieId (int): The index for the item
        """
        return self._data[self._data["movieId"] == movieId]["userId"].tolist()

    def list_items_for_rater(self, userId: int) -> list:
        """Returns a list of items rated by userId.
        Args:
            userId (int): The index for the user
        """
        return self._data[self._data["userId"] == userId]["movieId"].tolist()

    def summarize(self) -> pd.DataFrame:
        """Returns a dataet summary"""
        self._summarize()
        return self._summary

    def as_dto(self) -> pd.DataFrame:
        """Returns a Data Transfer Object representation of the entity."""
        return DatasetDTO(
            id=self._id,
            name=self._name,
            type=self._type,
            description=self._description,
            workspace=self._workspace,
            stage=self._stage,
            rows=self._rows,
            cols=self._cols,
            n_users=self._n_users,
            n_items=self._n_items,
            size=self._size,
            matrix_size=self._matrix_size,
            memory_mb=self._memory_mb,
            cost=self._cost,
            sparsity=self._sparsity,
            density=self._density,
            filepath=self._filepath,
        )

    def _summarize(self) -> None:
        """Sets a data profile including basic summary statistics"""
        if self._summary is None:
            # Computes basic statistics
            self._rows = int(self._data.shape[0])
            self._cols = int(self._data.shape[1])
            self._n_users = int(self._data["userId"].nunique())
            self._n_items = int(self._data["movieId"].nunique())
            self._size = int(self._nrows * self._ncols)
            self._matrix_size = int(self._n_users * self._n_items)
            self._sparsity = self._nrows / self._size * 100
            self._density = 100 - self._sparsity
            self._memory_mb = self._data.memory_usage(deep=True).sum() / 1024**2

            d = {}
            d["id"] = self._id
            d["name"] = self._name
            d["type"] = self._type
            d["description"] = self._description
            d["workspace"] = self._workspace
            d["stage"] = self._stage
            d["rows"] = self._rows
            d["cols"] = self._cols
            d["n_users"] = self._n_users
            d["n_items"] = self._n_items
            d["size"] = self._size
            d["matrix_size"] = self._matrix_size
            d["memory_mb"] = self._memory_mb
            d["cost"] = self._cost
            d["sparsity"] = self._sparsity
            d["density"] = self._density
            d["filepath"] = self._filepath

            self._summary = pd.DataFrame.from_dict(data=d, orient="index", columns=["Measure"])
