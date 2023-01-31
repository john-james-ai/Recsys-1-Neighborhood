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
# Modified   : Monday January 30th 2023 11:32:36 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Utility Matrix Module"""
from scipy import sparse

from recsys.io.file import IOService
from recsys.neighborhood.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class UtilityMatrix(Matrix):
    """Utility Matrix Contains User/Item Ratings in a Sparse Matrix Format

    Args:
        df_filepath (str): The filepath for the pandas DataFrame containing the user/item/rating data
        um_filepath (str): The filepath to which the utility matrix will be stored.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._matrix = None  # CSR Format
        self._load_data(filepath)

    @property
    def shape(self) -> tuple:
        """Returns tuple of the shape of the matrix"""
        return (
            self._df.shape[0],
            self._df.shape[1],
        )

    @property
    def size(self) -> int:
        """The number of cells in the matrix"""
        return self._df.shape[0] * self._df.shape[1]

    @property
    def memory(self) -> dict:
        """Memory consumed by matrix in bytes."""
        return self._matrix.data.nbytes

    def to_array(self) -> None:
        """Converts the matrix to numpy array format"""
        return self._matrix.toarray()

    def to_csr(self) -> None:
        """Converts a numpy matrix to compressed sparse row matrix"""
        return self._matrix

    def load(self) -> None:
        """Loads the matrix from file"""
        self._df = IOService.read(self._df_filepath)

    def save(self, filepath: str) -> None:
        """Saves the matrix to file"""
        try:
            self._um = IOService.read(self._um_filepath)
        except FileNotFoundError:
            self._logger.error()

    def _load_data(self, filepath: str) -> None:
        df = IOService.read(filepath)
        self._matrix = sparse.csr_matrix((df.rating, (df.userId, df.movieId)))
