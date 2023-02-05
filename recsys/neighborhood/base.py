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
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 09:08:48 pm                                                #
# Modified   : Sunday February 5th 2023 05:56:27 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base class for neighborhood collaborative filtering package"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Union

import numpy as np
from scipy import sparse
from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.io.repo import Repo


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
    def __call__(
        self,
        X: sparse.csr_matrix,
        Y: sparse.csr_matrix = None,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, sparse.csr_matrix]:
        """Computes the similarity between users"""

    def _normalize(
        self, X: Union[sparse.csr_matrix, sparse.csc_matrix], norm: str = "l2", axis: int = 1
    ) -> Union[sparse.csr_matrix, sparse.csc_matrix]:
        """Scales input vectors individually to unit norm.

        Args:
            X (sparse.csr_matrix, sparse.csc_matrix): The data to normalize
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
            X = sparse.csr_matrix.transpose(X)

        if norm == "l1":
            norms = abs(X).sum(axis=1)
        else:
            norms = np.sqrt(X.power(2).sum(axis=1))
        X = X / norms
        X = sparse.csr_matrix(X)  # Division by dense vector returns dense array
        if axis == 0:
            X = sparse.csr_matrix.transpose(X)
        return X

    def _check_input(
        self,
        X: Union[sparse.csc_matrix, sparse.csr_matrix],
        Y: Union[sparse.csc_matrix, sparse.csr_matrix],
    ) -> Union[
        tuple[sparse.csr_matrix, sparse.csr_matrix], tuple[sparse.csc_matrix, sparse.csc_matrix]
    ]:
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
        X: Union[sparse.csr_matrix, sparse.csc_matrix],
        none_allowed: bool = False,
    ) -> sparse.csr_matrix:
        """Checks array type, casts it to float

        Args:
            X (sparse.csr_matrix): Input array
        """
        if X is None and none_allowed:
            return X

        elif not isinstance(X, (sparse.csr_matrix, sparse.csc_matrix)):
            msg = f"Type {type(X)} is not supported. Must be 'np.ndarray' or 'sparse.csr_matrix'."
            self._logger.error(msg)
            raise TypeError(msg)

        elif not X.ndim == 2:
            msg = "Input must be a 2 dimensional array or sparse matrix."
            self._logger.error(msg)
            raise ValueError(msg)

        X = sparse.csr_matrix.asfptype(X)

        return X


# ------------------------------------------------------------------------------------------------ #
class Matrix(ABC):
    """Base class for recommender system matrices."""

    def __init__(self, name: str, *args, **kwargs) -> None:
        self._name = name
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def name(self) -> tuple:
        """Returns name of the matrix"""

    @property
    @abstractmethod
    def dataset(self) -> str:
        """The dataset name"""

    @property
    @abstractmethod
    def mean_centered(self) -> Union[str, bool]:
        """Indicates whether the ratings in the matrix have been mean centered"""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Returns tuple of the shape of the matrix"""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of cells in the matrix"""


# ------------------------------------------------------------------------------------------------ #
class MatrixFactory(ABC):
    @inject
    def __init__(self, repo: Repo = Provide[Recsys.repo.repo]) -> None:
        self._repo = repo
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Matrix:
        """Constructs the matrix"""
