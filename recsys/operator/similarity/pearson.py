#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/similarity/pearson.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 07:07:09 am                                                #
# Modified   : Friday March 3rd 2023 01:06:52 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pearson similarity module"""
from functools import cache

import pandas as pd
import numpy as np

from recsys.operator.similarity.base import UserSimilarityMeasure, ItemSimilarityMeasure


# ------------------------------------------------------------------------------------------------ #
class UserPearsonSimilarity(UserSimilarityMeasure):
    """User Pearson similarity measure

    Args:
        ratings_filepath (str): Path to the ratings file
        cooccurrence_filepath (str): Path to the file containing cooccurrence data
        destination (str): The path in which the similarity matrix is stored.
        uservar (str): Column containing user id
        itemvar (str): Column containing item id
        ratingvar (str): Column containing rating data
        force (bool): Whether to force execution of data already exists.

    """

    def __init__(
        self,
        ratings_filepath: str,
        cooccurrence_filepath,
        destination: str,
        uservar: str = "userId",
        itemvar: str = "movieId",
        ratingvar: str = "rating_ctr_user",
        force: bool = False,
    ) -> None:
        super().__init__(
            ratings_filepath=ratings_filepath,
            cooccurrence_filepath=cooccurrence_filepath,
            destination=destination,
            uservar=uservar,
            itemvar=itemvar,
            ratingvar=ratingvar,
            force=force,
        )

    @cache
    def _compute(self, pair: tuple, subjects: list, ratings: pd.DataFrame) -> float:
        """Computes the similarity between pairs of users or items."""
        u, v = pair
        # Obtain all ratings for user u for common items
        rui = ratings[(ratings[self._uservar] == u) & (ratings[self._itemvar].isin(subjects))][
            self._ratingvar
        ]

        # Obtain all ratings for user v for common items
        rvi = ratings[(ratings[self._uservar] == v) & (ratings[self._itemvar].isin(subjects))][
            self._ratingvar
        ]
        # Normalize ru and rv
        l2rui = np.sqrt(np.sum(np.square(rui)))
        l2rvi = np.sqrt(np.sum(np.square(rvi)))

        # Compute similarity
        sim = rui.dot(rvi) / (l2rui * l2rvi)
        return sim


# ------------------------------------------------------------------------------------------------ #
class ItemPearsonSimilarity(ItemSimilarityMeasure):
    """User Pearson similarity measure

    Args:
        ratings_filepath (str): Path to the ratings file
        cooccurrence_filepath (str): Path to the file containing cooccurrence data
        destination (str): The path in which the similarity matrix is stored.
        uservar (str): Column containing user id
        itemvar (str): Column containing item id
        ratingvar (str): Column containing rating data
        force (bool): Whether to force execution of data already exists.

    """

    def __init__(
        self,
        ratings_filepath: str,
        cooccurrence_filepath,
        destination: str,
        uservar: str = "userId",
        itemvar: str = "movieId",
        ratingvar: str = "rating_ctr_item",
        force: bool = False,
    ) -> None:
        super().__init__(
            ratings_filepath=ratings_filepath,
            cooccurrence_filepath=cooccurrence_filepath,
            destination=destination,
            uservar=uservar,
            itemvar=itemvar,
            ratingvar=ratingvar,
            force=force,
        )

    @cache
    def _compute(self, pair: tuple, subjects: list, ratings: pd.DataFrame) -> float:
        """Computes the similarity between pairs of users or items."""
        i, j = pair
        # Obtain all ratings for item i for common users
        riu = ratings[(ratings[self._itemvar] == i) & (ratings[self._uservar].isin(subjects))][
            self._ratingvar
        ]

        # Obtain all ratings for item j common users
        rju = ratings[(ratings[self._itemvar] == j) & (ratings[self._uservar].isin(subjects))][
            self._ratingvar
        ]

        # Normalize ru and rv
        l2riu = np.sqrt(np.sum(np.square(riu)))
        l2rju = np.sqrt(np.sum(np.square(rju)))

        # Compute similarity
        sim = riu.dot(rju) / (l2riu * l2rju)
        return sim
