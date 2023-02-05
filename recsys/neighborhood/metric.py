#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/metric.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 4th 2023 05:26:17 am                                              #
# Modified   : Saturday February 4th 2023 07:57:42 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union
from scipy import sparse
import numpy as np

from recsys.neighborhood.base import Metric


# ------------------------------------------------------------------------------------------------ #
class CosineSimilarity(Metric):
    """Creates a Cosine similarity matrix."""

    def __init__(self, name: str = "Cosine_similarity") -> None:
        super().__init__(name=name)

    def __call__(
        self,
        X: Union[np.ndarray, sparse.csr_matrix],
        Y: Union[np.ndarray, sparse.csr_matrix] = None,
        dimension: str = 'user',
        return_sparse: bool = True,
    ) -> Union[np.ndarray, sparse.csr_matrix]:
        """Computes Cosine similarity between samples X and Y

        Args:
            X (np.ndarray, sparse.csr_matrix): array or sparse matrix
            Y (np.ndarray, sparse.csr_matrix): array, sparse matrix or None
            return_sparse (bool): Whether to return an array or sparse matrix.

        """
        X, Y = self._check_input(X, Y)

        if isinstance(X, np.ndarray):
            C = self._cosine_arrays(X, Y)
        else:
            C = self._cosine_sparse(X, Y)

        if return_sparse and isinstance(C, np.ndarray):
            C = sparse.csr_matrix(C)

        return C

    def _cosine_arrays(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Computes Cosine similarity for numpy arrays"""
        XY = X.dot(Y.T)
        return self._normalize(XY, norm="l2", axis=1)

    def _cosine_sparse(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes Cosine similarity for sparse csr matrices."""
        XY = X.dot(Y.transpose())
        return self._normalize(XY, norm="l2", axis=1)
