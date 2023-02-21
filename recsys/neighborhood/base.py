#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/base.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 09:08:48 pm                                                #
# Modified   : Monday February 20th 2023 09:59:51 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base class for neighborhood collaborative filtering package"""
from __future__ import annotations
import sys
from abc import ABC, abstractmethod
import logging
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from recsys.dal.rating import RatingsDataset


# ------------------------------------------------------------------------------------------------ #
class Metric(ABC):
    """Base class for vectorized co-occurrence and similarity metric computations"""

    def __init__(self, name: str, **kwargs) -> None:
        self._name = name
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self, ratings: RatingsDataset, *args, **kwargs) -> Matrix:
        """Computes the similarity between users"""

    def _normalize(
        self, X: Union[csr_matrix, csc_matrix], norm: str = "l2", axis: int = 1
    ) -> Union[csr_matrix, csc_matrix]:
        """Scales input vectors individually to unit norm.

        Args:
            X (csr_matrix, csc_matrix): The data to normalize
            norm (str): One of ['l1', 'l2']. Default = 'l2'
            axis (int): Defines the axis along which the data are normalize. Either 0 (items) or 1 (users). Default = 1
        """
        X = self._check_matrix(X)

        if norm not in ["l1", "l2"]:
            msg = f"Norm {norm} is not supported."
            self._logger.error(msg)
            raise ValueError(msg)
        if axis not in [0, 1]:
            msg = "Axis must be in [0,1]."
            self._logger.error(msg)
            raise ValueError(msg)

        if axis == 0:
            X = csr_matrix.transpose(X)

        if norm == "l1":
            norms = abs(X).sum(axis=1)
        else:
            norms = np.sqrt(X.power(2).sum(axis=1))
        X = X / norms
        X = csr_matrix(X)  # Division by dense vector returns dense array
        if axis == 0:
            X = csr_matrix.transpose(X)
        return X

    def _check_input(
        self,
        X: Union[csc_matrix, csr_matrix],
        Y: Union[csc_matrix, csr_matrix],
    ) -> Union[tuple[csr_matrix, csr_matrix], tuple[csc_matrix, csc_matrix]]:
        """Checks type and dimension of input."""
        X = self._check_matrix(X)
        Y = self._check_matrix(Y, none_allowed=True)
        Y = Y or X

        # Forcing input to be same type, because I ain't got no time for that sh&*.
        if type(X) != type(Y):
            msg = f"X: type {type(X)} and Y: type {type(Y)} must be the same type."
            self._logger.error(msg)
            raise TypeError(msg)

        if not all(np.equal(X.shape, Y.shape)):
            msg = f"Shape mismatch: X.shape = {X.shape}: Y.shape = {Y.shape}."
            self._logger.error(msg)
            raise ValueError(msg)
        return (
            X,
            Y,
        )

    def _check_matrix(
        self,
        X: Union[csr_matrix, csc_matrix],
        none_allowed: bool = False,
    ) -> csr_matrix:
        """Checks array type, casts it to float

        Args:
            X (csr_matrix): Input array
        """
        if X is None and none_allowed:
            return X

        elif not isinstance(X, (csr_matrix, csc_matrix)):
            msg = f"Type {type(X)} is not supported. Must be 'np.ndarray' or 'csr_matrix'."
            self._logger.error(msg)
            raise TypeError(msg)

        elif not X.ndim == 2:
            msg = "Input must be a 2 dimensional array or sparse matrix."
            self._logger.error(msg)
            raise ValueError(msg)

        X = csr_matrix.asfptype(X)

        return X


# ------------------------------------------------------------------------------------------------ #
class Matrix(ABC):
    def __init__(
        self, matrix: Union[csr_matrix, csc_matrix], ratings: RatingsDataset, user: bool = True
    ) -> None:
        self._matrix = matrix
        self._mode = ratings.mode
        self._ratings = ratings.name
        self._dataset = ratings.dataset
        self._user = user
        self._name = None

    @property
    def name(self) -> int:
        """Returns the name by which the object will be persisted."""
        return self._name

    @property
    def size(self) -> int:
        return self._matrix.data.nbytes

    @property
    def nnz(self) -> int:
        return self._matrix.nnz

    @property
    def mode(self) -> int:
        """Returns the mode in qhich the index was created."""
        return self._mode

    @property
    def ratings(self) -> str:
        """Returns the name of the ratings object, from which the index derives."""
        return self._ratings

    @property
    def dataset(self) -> str:
        """Returns the name of the dataset, i.e. 'train', 'test'."""
        return self._dataset


# ------------------------------------------------------------------------------------------------ #
class Index(ABC):
    def __init__(self, index: dict, ratings: RatingsDataset, user: bool = True) -> None:
        self._index = index
        self._mode = ratings.mode
        self._ratings = ratings.name
        self._dataset = ratings.dataset
        self._user = user

    def __len__(self) -> int:
        """Total number of items in the index"""
        return len(self._index)

    @property
    def name(self) -> int:
        """Returns the name by which the object will be persisted."""
        return self._name

    @property
    def mode(self) -> int:
        """Returns the mode in qhich the index was created."""
        return self._mode

    @property
    def ratings(self) -> str:
        """Returns the name of the ratings object, from which the index derives."""
        return self._ratings

    @property
    def dataset(self) -> str:
        """Returns the name of the dataset, i.e. 'train', 'test'."""
        return self._dataset

    @property
    def size(self) -> int:
        """Total size of index in memory"""
        return sys.getsizeof(self._index)

    @abstractmethod
    def search(self, **kwargs) -> list:
        """Returns a list of items or users matching search criteria"""


# ------------------------------------------------------------------------------------------------ #
class IndexFactory(ABC):
    def __init__(self, ratings: RatingsDataset) -> None:
        self._ratings = ratings
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def create_user(self) -> Index:
        """Creates the user version of the index"""

    @abstractmethod
    def create_item(self) -> Index:
        """Creates the item version of the index"""
