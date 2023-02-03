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
# Modified   : Thursday February 2nd 2023 06:19:26 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Utility Matrix Module"""
import pandas as pd

from recsys.neighborhood.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(Matrix):
    """Similarity Matrix Contains User/Item Ratings in a Sparse Matrix Format

    Args:
        name
        um_filepath (str): The filepath to which the utility matrix will be stored.
    """

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        super().__init__(name=name, data=data)

    @property
    def name(self) -> tuple:
        """Returns name of the matrix"""
        return self._name

    @property
    def shape(self) -> tuple:
        """Returns tuple of the shape of the matrix"""
        return (
            self._data.shape[0],
            self._data.shape[1],
        )

    @property
    def size(self) -> int:
        """The number of cells in the matrix"""
        return self._data.shape[0] * self._data.shape[1]

    @property
    def memory(self) -> dict:
        """Memory consumed by matrix in bytes."""
        return self._data.memory_usage(deep=True).sum()

    def load(self) -> None:
        """Loads the matrix from file"""
        self._data = self._repo.get(self._name)

    def save(self) -> None:
        """Saves the matrix to file"""
        if self._repo.exists(name=self._name):
            self._repo.update(name=self._name, item=self._data)
        else:
            self._repo.add(name=self._name, item=self._data)
