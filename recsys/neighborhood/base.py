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
# Modified   : Friday February 3rd 2023 11:04:24 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base class for neighborhood collaborative filtering package"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations

from dependency_injector.wiring import inject, Provide

from recsys.data.rating import RatingsDataset
from recsys.container import Recsys


# ------------------------------------------------------------------------------------------------ #
class SimilarityMetric(ABC):
    """Base class for similarity metrics"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )
        self._user_similarity = None
        self._item_similarity = None

    @property
    def user_similarity(self) -> Matrix:
        return self._user_similarity

    @property
    def item_similarity(self) -> Matrix:
        return self._item_similarity

    @abstractmethod
    def compute_user_similarity(self, ratings: RatingsDataset) -> Matrix:
        """Computes the similarity between users"""

    @abstractmethod
    def compute_item_similarity(self, ratings: RatingsDataset) -> Matrix:
        """Computes the similarity between items"""

    def _extract_common_items(self, ratings: RatingsDataset) -> dict:
        """For each item, list pairs of users who have rated the item"""
        Iuv = {}
        # Obtain all unique items sorted
        for item in tqdm(ratings.items):
            # Extract all ratings for the item
            item_ratings = ratings.get_item_ratings(item=item)
            # Iterate over each pair of users who rated the item
            for uv_pair in combinations(item_ratings["userId"].values, 2):
                # Extract the item's ratings for the pair of users
                uv_ratings = item_ratings[item_ratings["userId"].isin(uv_pair)]
                uv_key = self._to_key(uv_pair)
                # Add the ratings for the pair to inverse index of items u,v user ratings
                if Iuv.get(uv_key, None) is not None:
                    Iuv[uv_key] = pd.concat([Iuv[uv_key], uv_ratings], axis=0)
                else:
                    Iuv[uv_key] = uv_ratings
        return Iuv

    def _extract_common_users(self, ratings: RatingsDataset) -> dict:
        """For each user, list the pairs of items the user has rated."""
        # Obtain all unique users sorted
        Uij = {}
        for user in tqdm(ratings.users):
            # Extract all ratings for the user
            user_ratings = ratings.get_user_ratings(user=user)
            # Iterate over each pair of items rated by the user
            for ij_pair in combinations(user_ratings["movieId"].values, 2):
                # Extract the user's ratings for the pair of items
                ij_ratings = user_ratings[user_ratings["movieId"].isin(ij_pair)]
                ij_key = self._to_key(ij_pair)
                # Add the ratings for the pair to inverse index of user i,j item ratings
                if Uij.get(ij_key, None) is not None:
                    Uij[ij_key] = pd.concat([Uij[ij_key], ij_ratings], axis=0)
                else:
                    Uij[ij_key] = ij_ratings
        return Uij

    def _compute_user_rating_norms(self, ratings: RatingsDataset, centered_by: str = None):
        """Computes L2 Norm for user ratings that have optionally been centered."""
        ru = ratings.get_user_ratings(centered_by=centered_by)
        return np.sqrt(np.sum(np.square(ru)))

    def _compute_item_rating_norms(self, ratings: RatingsDataset, centered_by: str = None):
        """Computes L2 Norm for user ratings that have optionally been centered."""
        ri = ratings.get_item_ratings(centered_by=centered_by)
        return np.sqrt(np.sum(np.square(ri)))

    def _to_key(self, t: tuple) -> str:
        """Returns a string dictionary key for a tuple"""
        return str(t[0]) + "_" + str(t[1])

    def _from_key(self, s: str) -> tuple:
        """Returns a tuple for a string dictionary key"""
        return tuple([int(x) for x in s.split("_")])


# ------------------------------------------------------------------------------------------------ #
class Matrix:
    """Base class for recommender system matrices."""

    @inject
    def __init__(self, name: str, data: pd.DataFrame, repo=Provide[Recsys.repo]) -> None:
        self._name = name
        self._data = data
        self._repo = repo
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def name(self) -> tuple:
        """Returns name of the matrix"""

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
    def load(self) -> None:
        """Loads the matrix from file."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Saves the matrix to file"""
