#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/factory.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 4th 2023 02:51:47 pm                                              #
# Modified   : Saturday February 4th 2023 09:10:24 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Matrix Factory Module"""
from itertools import combinations

from tqdm import tqdm

from recsys.data.rating import RatingsDataset
from recsys.neighborhood.base import MatrixFactory, Metric
from recsys.neighborhood.matrix import SimilarityMatrix, InvertedIndex


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrixFactory(MatrixFactory):
    """Similarity matrix"""

    def __init__(self, ratings: RatingsDataset) -> None:
        super().__init__()
        self._ratings = ratings

    def __call__(self, name: str, metric: Metric, dimension: str = "user") -> SimilarityMatrix:
        """Creates a similarity matrix using the metric object

        Args:
            name (str): Name of the SimilarityMatrix object
            metric (Type[Metric]): Metric type object.
            dimension (str): ['user', 'item']. The dimension over which similarity is computed. Default='user'
        """
        X = self._ratings.as_csr()
        C = metric(X, return_sparse=True)

        matrix = SimilarityMatrix(name=name, matrix=C, dimension=dimension)
        self._repo.add(name=name, item=matrix)
        return matrix


# ------------------------------------------------------------------------------------------------ #
class InvertedIndexFactory(MatrixFactory):
    """Constructs user and item co-occurrence inverted index objects"""

    def __init__(self, ratings: RatingsDataset) -> None:
        super().__init__()
        self._ratings = ratings

    def __call__(self, name: str, dimension: str = "user") -> InvertedIndex:
        """Constructs an inverted index of co-occurring elements within the dimension

        Args:
            name (str): Name of the SimilarityMatrix object
            dimension (str): ['user', 'item']. The dimension over which similarity is computed. Default='user'
        """
        if "u" in dimension:
            index = self._build_user_index()
        else:
            index = self._build_item_index()

        ii = InvertedIndex(name=name, index=index, dimension=dimension)
        self._repo.add(name=name, item=ii)
        return ii

    def _build_user_index(self) -> dict:
        """Constructs index of co-occuring users, i.e. rated same film."""
        UV = {}

        for item in tqdm(self._ratings.items):
            item_ratings = self._ratings.get_item_ratings(item=item)
            for uv_pair in combinations(item_ratings["useridx"].values, 2):
                if UV.get(uv_pair, None) is not None:
                    UV[uv_pair].append(item)
                else:
                    UV[uv_pair] = [item]
        return UV

    def _build_item_index(self) -> dict:
        """Constructs index of co-occuring items, i.e. rated by a user."""
        IJ = {}

        for user in tqdm(self._ratings.users):
            user_ratings = self._ratings.get_user_ratings(user=user)
            for ij_pair in combinations(user_ratings["itemidx"].values, 2):
                if IJ.get(ij_pair, None) is not None:
                    IJ[ij_pair].append(user)
                else:
                    IJ[ij_pair] = [user]
        return IJ
