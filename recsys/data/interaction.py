#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/data/interaction.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 6th 2023 03:30:16 am                                                   #
# Modified   : Thursday March 9th 2023 04:15:23 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Interaction Matrix Module"""
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import pandas as pd
from typing import Union

from recsys.data.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class Interaction(Matrix):
    """Interaction Class"""

    def __init__(self, rows: 160000, cols: 60000, umax: int = 1000, imax: int = 1000) -> None:
        super().__init__()
        self._dataset = None
        self._user_counts = None
        self._item_counts = None
        self._umax = umax
        self._imax = imax
        self._csr = None
        self._csc = None
        self._is_array = True
        self._array = None
        self._total_user_interactions = np.zeros(shape=imax)
        self._total_item_interactions = np.zeros(shape=umax)
        self._initialize()

    def _initialize(self) -> None:
        self._array = np.zeros(shape=(self._umax, self._imax))

    @property
    def rows(self) -> int:
        """Returns the number of rows."""

    @property
    def cols(self) -> int:
        """Returns the number of columns."""

    @property
    def size(self) -> int:
        """Returns the matrix size."""

    @property
    def nnz(self) -> int:
        """Returns the number of non-zero elements."""

    def add(self, batch: pd.DataFrame) -> None:
        """Adds a batch of interactions to the interaction matrix."""
        batch == batch[["userId", "movieId", "timestamp"]]
        batch["interaction"] == 1

    def get(self, row: int, col: int) -> Union[int, float]:
        """Retunrs the element at specified row and column."""

    def dot(self, other: Matrix) -> Matrix:
        """Performs a dot product with an other matrix"""

    def sum(self, other: Matrix) -> Matrix:
        """Adds an other matrix object."""

    def to_numpy(self) -> None:
        """Return the 2 dimensional numpy array"""
        if self._is_array:
            pass
        elif self._csr:
            self._array = self._csr.toarray()
        elif self._csc:
            self._array = self._csc.toarray()
        return self._array

    def to_csc(self) -> csc_matrix:
        """Return the sparse csc matrix."""
        self._csr = csc_matrix(self._umax, self._imax)

    def to_csr(self) -> csr_matrix:
        """Return the sparse csr matrix."""
        self._csr = csr_matrix(self._umax, self._imax)

    def _create_binary_matrix(self, batch: pd.DataFrame) -> np.ndarray:
        """Convert batch to binary matrix"""
        self._create_binary_matrix()
        row = self._batch["userId"].values
        col = self._batch["movieId"].values
        data = self._batch["rating"].values
        csr = csr_matrix((data, (row, col)), shape=(self._imax, self._umax))
        array = csr.toarray()
        array = np.where(array > 0, 1, 0)
        return array
