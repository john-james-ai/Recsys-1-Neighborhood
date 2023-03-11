#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/data/matrix.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 6th 2023 03:30:16 am                                                   #
# Modified   : Thursday March 9th 2023 06:42:34 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Interaction Matrix Module"""
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import numpy as np

from recsys.data.base import MatrixABC


# ------------------------------------------------------------------------------------------------ #
class Matrix(MatrixABC):
    """Matrix Class"""

    def __init__(
        self, name: str, description: str, data: coo_matrix, datasource: str = "movielens25m"
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._datasource = datasource
        self._coo = data

        self._csr = None
        self._csc = None

    @property
    def rows(self) -> int:
        """Returns the number of rows."""
        return self._coo.shape[0]

    @property
    def cols(self) -> int:
        """Returns the number of columns."""
        return self._coo.shape[1]

    @property
    def size(self) -> int:
        """Returns the matrix size as nrows * ncols."""
        return self._coo.shape[0] * self._coo.shape[1]

    @property
    def nnz(self) -> int:
        """Returns the number of non-zero elements."""
        return self._coo.nnz

    def sum(self, axis: int) -> np.array:
        """Returns sum along an axis."""
        return self._coo.sum(axis=axis)

    def getrow(self, row: int) -> np.array:
        """Returns the designated row."""
        return self._coo.getrow(i=row)

    def getcol(self, col: int) -> np.array:
        """Returns the designated column."""
        return self._coo.getcol(j=col)

    def to_numpy(self) -> None:
        """Return the 2 dimensional numpy array"""
        return self._coo.toarray()

    def to_csc(self) -> csc_matrix:
        """Return the sparse csc matrix."""
        if self._csc is None:
            self._csc = self._coo.tocsc()
        return self._csc

    def to_csr(self) -> csr_matrix:
        """Return the sparse csr matrix."""
        if self._csr is None:
            self._csr = self._coo.tocsr()
        return self._csr
