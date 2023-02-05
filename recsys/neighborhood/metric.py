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
# Modified   : Sunday February 5th 2023 07:42:12 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from scipy import sparse

from recsys.neighborhood.base import Metric


# ------------------------------------------------------------------------------------------------ #
class CosineSimilarity(Metric):
    """Creates a Cosine similarity matrix."""

    def __init__(self, name: str = "cosine_similarity") -> None:
        super().__init__(name=name)

    def __call__(
        self, X: sparse.csr_matrix, Y: sparse.csr_matrix = None, dimension: str = "user"
    ) -> sparse.csr_matrix:
        """Computes Cosine similarity between samples X and Y

        Args:
            X (sparse.csr_matrix): array or sparse matrix
            Y (sparse.csr_matrix): array, sparse matrix or None
            dimension (str): Either 'user' or 'item'

        """
        X, Y = self._check_input(X, Y)
        if "u" in dimension:
            return self._cosine_user(X, Y)
        else:
            return self._cosine_item(X, Y)

    def _cosine_user(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes user cosign similarity, normalizing over all items"""
        XY = X.dot(Y.transpose())
        return self._normalize(XY, norm="l2", axis=1)

    def _cosine_item(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes item cosign similarity, normalizing over all users"""
        XY = X.transpose().dot(Y)
        return self._normalize(XY, norm="l2", axis=0)


# ------------------------------------------------------------------------------------------------ #
class PearsonSimilarity(Metric):
    """Creates a Pearson similarity matrix.

    Computes the pairwise user or item similarities considering only ratings for the
    common items.
    """

    def __init__(self, coname: str = "pearson_similarity") -> None:
        super().__init__(name=name)

    def __call__(
        self, X: sparse.csr_matrix, Y: sparse.csr_matrix = None, dimension: str = "user"
    ) -> sparse.csr_matrix:
        """Computes Cosine similarity between samples X and Y

        Args:
            X (sparse.csr_matrix): array or sparse matrix
            Y (sparse.csr_matrix): array, sparse matrix or None
            dimension (str): Either 'user' or 'item'

        """
        X, Y = self._check_input(X, Y)
        if "u" in dimension:
            return self._pearson_user(X, Y)
        else:
            return self._pearson_item(X, Y)

    def _pearson_user(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes user cosign similarity, normalizing over all items"""
        XY = X.dot(Y.transpose())
        return self._normalize(XY, norm="l2", axis=1)

    def _pearson_item(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes item cosign similarity, normalizing over all users"""
        XY = X.transpose().dot(Y)
        return self._normalize(XY, norm="l2", axis=0)
