#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/similarity.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 05:01:09 am                                                #
# Modified   : Monday March 6th 2023 01:27:49 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Similarity Matrix Module"""

import pandas as pd

from recsys.data.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(Dataset):
    """Similarity Matrix

    Args:
        name (str): The name of the similarity matrix
        description (str): Describes the similarity matrix in terms of items or users
        data (pd.DataFrame): The similarity matrix data in DataFrame format.
        measure (SimilarityMeasure): The SimilarityMeasure object that was used.
        dimension (str): Either ['userId', or 'movieid'] for user similarity or item similarity,
            respectively.
    """

    def __init__(
        self, name: str, description: str, data: pd.DataFrame, measure: str, dimension: str
    ) -> None:
        super().__init__(name=name, description=description, data=data)
        self._measure = measure
        self._dimension = dimension

    @property
    def measure(self) -> str:
        return self._measure

    def get_similarity(self, a: int, b: int) -> float:
        """Returns the similarity measure for a pair of users or items.

        Args:
            a (int): Either a user or item
            b (int): Either a user or item, matching type of a.
        """
        if a >= b:  # Swap the values as similarity measures are calculated in u,v order.
            c = a
            a = b
            b = c
        return self._dataframe[(self._dataframe["a"] == a) & (self._dataframe["b"] == b)]["score"]
