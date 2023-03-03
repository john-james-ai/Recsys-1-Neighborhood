#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/similarity/base.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 05:05:32 am                                                #
# Modified   : Friday March 3rd 2023 01:00:33 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import abstractmethod

import pandas as pd

from recsys.operator.base import Operator


# ------------------------------------------------------------------------------------------------ #
class SimilarityMeasure(Operator):
    """Base class for Similarity Methods

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
        ratingvar: str = "rating",
        force: bool = False,
    ) -> None:
        super().__init__(destination=destination, force=force)
        self._ratings_filepath = ratings_filepath
        self._cooccurrence_filepath = cooccurrence_filepath
        self._uservar = uservar
        self._itemvar = itemvar
        self._ratingvar = ratingvar

    @abstractmethod
    def execute(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Computes the similarity measure"""


# ------------------------------------------------------------------------------------------------ #
class UserSimilarityMeasure(SimilarityMeasure):
    """Base class for User Similarity Measures

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
        ratingvar: str = "rating",
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

    def execute(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Computes the similarity measure"""
        similarity = []

        if not self._skip(endpoint=self._destination):
            # Obtain the ratings and cooccurrence data
            ratings = self._get_data(filepath=self._ratings_filepath)
            cooccurrence = self._get_data(filepath=self._cooccurrence_filepath)
            # Iterate over a data grouped by the cooccurrents
            for user_pair, items in cooccurrence.groupby(self._uservar):
                # Compute similarity over the pair
                similarity = self._compute(
                    pair=user_pair, subjects=items[self._itemvar].values, ratings=ratings
                )
                d = {self._uservar: user_pair, "similarity": similarity}
                similarity.append(d)
            df = pd.DataFrame(data=similarity)

            # Split tuple column
            df[["u", "v"]] = pd.DataFrame(df[self._uservar].tolist(), index=df.index)
            df.drop(columns=[self._uservar], inplace=True)

            self._put_data(filepath=self._destination, data=df)

    @abstractmethod
    def _compute(self, pair: tuple, subjects: list, ratings: pd.DataFrame) -> float:
        """Computes the similarity between pairs of users or items."""


# ------------------------------------------------------------------------------------------------ #
class ItemSimilarityMeasure(Operator):
    """Base class for User Similarity Measures

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
        ratingvar: str = "rating",
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

    def execute(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Computes the similarity measure"""
        similarity = []

        if not self._skip(endpoint=self._destination):
            # Obtain the ratings and cooccurrence data
            ratings = self._get_data(filepath=self._ratings_filepath)
            cooccurrence = self._get_data(filepath=self._cooccurrence_filepath)
            # Iterate over a data grouped by the cooccurrents
            for item_pair, users in cooccurrence.groupby(self._itemvar):
                # Compute similarity over the pair
                similarity = self._compute(
                    pair=item_pair, subjects=users[self._uservar].values, ratings=ratings
                )
                d = {self._itemvar: item_pair, "similarity": similarity}
                similarity.append(d)
            df = pd.DataFrame(data=similarity)

            # Split tuple column
            df[["i", "j"]] = pd.DataFrame(df[self._itemvar].tolist(), index=df.index)
            df.drop(columns=[self._itemvar], inplace=True)

            self._put_data(filepath=self._destination, data=df)

    @abstractmethod
    def _compute(self, pair: tuple, subjects: list, ratings: pd.DataFrame) -> float:
        """Computes the similarity between pairs of users or items."""
