#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/core/dataset.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 07:10:48 pm                                             #
# Modified   : Wednesday February 22nd 2023 11:34:02 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from recsys.core.base import Entity
from recsys.dal.dto import DatasetDTO


# ------------------------------------------------------------------------------------------------ #
class Dataset(Entity):
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
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id: int) -> int:
        self._id = id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def users(self) -> np.array:
        """Returns array of unique users"""
        return np.sort(self._data["userId"].unique())

    @property
    def items(self) -> np.array:
        """Returns array of unique items"""
        return np.sort(self._data["movieId"].unique())

    @property
    def rows(self) -> int:
        """Return the numbrr of rows."""
        return self._rows

    @property
    def cols(self) -> int:
        """Return the numbrr of columns."""
        return self._cols

    @property
    def n_users(self) -> int:
        """Return the number of users."""
        return self._n_used

    @property
    def n_items(self) -> int:
        """Return the numbrr of movies."""
        return self._n_items

    @property
    def size(self) -> int:
        """Return the numbrr of elements in the DataFrame."""
        return self._size

    @property
    def matrix_size(self) -> int:
        """The product of unique elements along each axis."""
        return self._matrix_size

    @property
    def sparsity(self) -> int:
        """Computed on main dataset"""
        return self._sparsity

    @property
    def density(self) -> int:
        """The inverse of sparsity."""
        return self._density

    @property
    def memory_mb(self) -> int:
        """Memory usage"""
        return self._memory_mb

    @property
    def cost(self) -> str:
        "The number of seconds to create the document."
        return self._cost

    @cost.setter
    def cost(self, cost: str) -> None:
        """Cost to produce the artifact."""
        self._cost = cost

    @property
    def filepath(self) -> str:
        """Location where the file is persisted."""
        return self._filepath

    @filepath.setter
    def filepath(self, filepath: str) -> None:
        """It will be set soon."""
        self._filepath = filepath

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
            lab=self._lab,
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

    def _get_lab(self) -> str:
        """Reads the current lab from the environment variable."""
        load_dotenv()
        return os.getenv("WORKSPACE", "dev")

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
            d["lab"] = self._lab
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
