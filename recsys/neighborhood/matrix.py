#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/matrix.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 08:37:49 pm                                                #
# Modified   : Saturday February 4th 2023 10:27:00 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Matrix Module"""
import sys
from scipy.sparse import csr_matrix

from recsys.neighborhood.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(Matrix):
    def __init__(self, name: str, matrix: csr_matrix, dimension: str = "user") -> None:
        super().__init__(name=name)
        self._matrix = matrix
        self._dimension = dimension

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> str:
        return self._dimension

    @property
    def shape(self) -> tuple:
        return self._matrix.shape

    @property
    def size(self) -> int:
        return self._matrix.size

    def as_csr(self) -> csr_matrix:
        return self._matrix.copy()


# ------------------------------------------------------------------------------------------------ #
class InvertedIndex(Matrix):
    """Index of users and items that co-occur as ratings.

    Args:
        name (str): Name of the index
        index (dict): Dictionary of lists containing co-occurring pairs of users or items
        dimension (str): Either 'user' or 'item'
    """

    def __init__(self, name: str, index: dict, dimension: str = "user") -> None:
        self._name = name
        self._index = index
        self._dimension = dimension

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> str:
        return self._dimension

    @property
    def shape(self) -> int:
        return len(self._index)

    @property
    def size(self) -> int:
        return sys.getsizeof(self._index)

    def get_pairs(self) -> list:
        """Returns a list of tuples. Each tuple contains a co-occurring pair of user or item identifiers"""
        return list(self._index.keys())

    def get_common_elements(self, a: int, b: int) -> list:
        """Returns a list of elements common to a and b.

        Args:
            a (int): an identifier for an element in the user or item dimension
            b (int): an identifier for an element in the user or item dimension
        """
        return self._index[(a, b)]
