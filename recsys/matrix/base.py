#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/matrix/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 06:33:06 pm                                                  #
# Modified   : Friday March 17th 2023 08:16:50 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Matrix Module"""
from abc import ABC, abstractmethod
import logging

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


# ------------------------------------------------------------------------------------------------ #
class Matrix(ABC):
    """Base class for sparse matrices."""
    def __init__(self) -> None:


    @property
    @abstractmethod
    def name(str) -> str:
        """Returns the name of the matrix object"""

    @property
    @abstractmethod
    def description(str) -> str:
        """Returns the description of the matrix object"""

    @property
    @abstractmethod
    def filepath(str) -> str:
        """Returns the filepath of the matrix object"""

    @property
    @abstractmethod
    def rows(self) -> int:
        """Returns the number of rows."""

    @property
    @abstractmethod
    def cols(self) -> int:
        """Returns the number of columns."""

    @property
    @abstractmethod
    def shape(self) -> int:
        """Returns the shape of the matrix."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Returns the matrix size as nrows * ncols."""

    @property
    @abstractmethod
    def nnz(self) -> int:
        """Returns the number of non-zero elements."""

    @abstractmethod
    def sum(self, axis: int) -> np.array:
        """Returns sum along an axis."""

    @abstractmethod
    def getrow(self, row: int) -> np.array:
        """Returns the designated row."""

    @abstractmethod
    def getcol(self, col: int) -> np.array:
        """Returns the designated column."""

    @abstractmethod
    def get_element(self, row: int, col: int) -> float:
        """Returns the element at the nonzero row and column."""

    @abstractmethod
    def to_numpy(self) -> np.array:
        """Return the 2 dimensional dense numpy array"""

    @abstractmethod
    def to_csc(self) -> csc_matrix:
        """Return the sparse csc matrix."""

    @abstractmethod
    def to_csr(self) -> csr_matrix:
        """Return the sparse csr matrix."""
