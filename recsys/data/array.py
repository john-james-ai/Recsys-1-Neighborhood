#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/data/array.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 6th 2023 03:30:16 am                                                   #
# Modified   : Saturday March 11th 2023 10:46:32 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Interaction Matrix Module"""
import numpy as np

from recsys.data.base import ArrayABC


# ------------------------------------------------------------------------------------------------ #
class Array(ArrayABC):
    """Array Class instantiated wth a coo_array"""

    def __init__(
        self, name: str, description: str, data: np.array, datasource: str = "movielens25m"
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._datasource = datasource
        self._array = data

    @property
    def shape(self) -> int:
        """Returns the shape of the matrix."""
        return self._array.shape

    @property
    def size(self) -> int:
        """Returns the matrix size as nrows * ncols."""
        return len(self._array)

    @property
    def min(self) -> int:
        """Returns the minimum in the array."""
        return self._array.min()

    @property
    def max(self) -> int:
        """Returns the maximum in the array."""
        return self._array.max()

    def to_numpy(self) -> None:
        """Return the 2 dimensional numpy array"""
        return self._array
