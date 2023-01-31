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
# Modified   : Monday January 30th 2023 10:32:30 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base class for neighborhood collaborative filtering package"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging


# ------------------------------------------------------------------------------------------------ #
class Similarity(ABC):
    """Base class for similarity measures"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Computes the similarity between users u, and v"""


# ------------------------------------------------------------------------------------------------ #
class Matrix:
    """Base class for recommender system matrices."""

    def __init__(self, *args, **kwargs) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Returns tuple of the shape of the matrix"""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of cells in the matrix"""

    @property
    @abstractmethod
    def memory(self) -> dict:
        """Memory consumed by matrix in bytes."""

    @abstractmethod
    def to_csr(self) -> None:
        """Returns matrix to compressed sparse row matrix format"""

    @abstractmethod
    def to_array(self) -> None:
        """Returns the matrix to numpy array format, if not already."""

    @abstractmethod
    def load(self) -> None:
        """Loads the matrix from file."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Saves the matrix to file"""

    @abstractmethod
    def dot(self, m: Matrix) -> Matrix:
        """Performs Dot Product of matrix with m"""
