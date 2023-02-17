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
# Modified   : Thursday February 16th 2023 11:05:35 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from scipy import sparse
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import tqdm

from recsys.neighborhood.base import Metric
from recsys.neighborhood.matrix import SimilarityMatrix
from recsys.data.rating import RatingsDataset


# ------------------------------------------------------------------------------------------------ #
class CosineSimilarity(Metric):
    """Creates a Cosine similarity matrix."""

    def __init__(self, name: str = "cosine_similarity") -> None:
        super().__init__(name=name)

    def __call__(self, ratings: RatingsDataset, dimension: str = "user") -> SimilarityMatrix:
        """Computes Cosine similarity between samples X and Y

        Args:
            ratings (RatingsDataset): ratings dataset object
            dimension (str): Either 'user' or 'item'

        """
        if "u" in dimension:
            return self._cosine_user(ratings)
        else:
            return self._cosine_item(ratings)

    def _cosine_user(self, ratings: RatingsDataset) -> SimilarityMatrix:
        """Computes user cosign similarity, normalizing over all items"""
        A = ratings.as_csc(axis=0)
        B = ratings.as_csc(axis=1)
        UU = A.dot(B)
        return self._normalize(UU, norm="l2", axis=1)

    def _cosine_item(self, ratings: RatingsDataset) -> SimilarityMatrix:
        """Computes item cosign similarity, normalizing over all users"""
        A = ratings.as_csr(axis=1)
        B = ratings.as_csr(axis=0)
        II = A.dot(B)
        return self._normalize(II, norm="l2", axis=1)


# ------------------------------------------------------------------------------------------------ #
class PearsonSimilarity(Metric):
    """Creates a Pearson similarity matrix.

    Computes the pairwise user or item similarities considering only ratings for the
    common items.
    """

    def __init__(self, name: str = "pearson_similarity") -> None:
        super().__init__(name=name)

    def __call__(
        self, X: sparse.csr_matrix, Y: sparse.csr_matrix = None, dimension: str = "user"
    ) -> sparse.csr_matrix:
        """Computes Pearson similarity between samples X and Y

        Args:
            X (sparse.csr_matrix): sparse matrix
            Y (sparse.csr_matrix): sparse matrix or None
            dimension (str): Either 'user' or 'item'

        """
        X, Y = self._check_input(X, Y)
        if "u" in dimension:
            return self._pearson_user(X, Y)
        else:
            return self._pearson_item(X, Y)

    def _pearson_user(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes user pearson similarity, normalizing only over common items"""
        X = sparse.csr_matrix.tocsc(X)
        users = list(X.indices)
        pairs = combinations(users, 2)
        u = []
        v = []
        s = []
        for pair in tqdm(pairs):
            # Get ratings for user u and v
            ru = X[pair[0], :].todense()[0, :].squeeze()
            rv = X[pair[1], :].todense()[0, :].squeeze()

            # Format boolean mask of common items Iuv and obtain Iu and Iv
            Iuv = np.logical_and(ru, rv)
            Iu = np.multiply(Iuv, ru)
            Iv = np.multiply(Iuv, rv)
            # Compute pearsons correlation
            pc = np.multiply(ru, rv).sum() / (
                np.sqrt(np.sum(np.square(Iu))) * np.sqrt(np.sum(np.square(Iv)))
            )
            # Add data to list
            u.append(pair[0])
            v.append(pair[1])
            s.append(pc)

        # Convert similarity matrix to csr format
        d = {"u": u, "v": v, "s": s}
        S = pd.DataFrame(d)
        row = S.u.values
        col = S.v.values
        size = S[["u", "v"]].max().max() + 1
        data = S.s.values
        csr = sparse.csr_matrix((data, (row, col)), shape=(size, size))
        return csr

    def _pearson_item(self, X: sparse.csr_matrix, Y: sparse.csr_matrix) -> sparse.csr_matrix:
        """Computes item pearson similarity, normalizing only on common users"""
        XY = X.transpose().dot(Y)
        return self._normalize(XY, norm="l2", axis=0)
