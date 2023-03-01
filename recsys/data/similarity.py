#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/similarity.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 05:01:09 am                                                #
# Modified   : Wednesday March 1st 2023 06:24:02 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Similarity Matrix Module"""

import pandas as pd

from recsys.assets.data import DataAsset
from recsys.operator.similarity.base import SimilarityMeasure


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(DataAsset):
    """Similarity Matrix

    Args:
        name (str): The name of the similarity matrix
        description (str): Describes the similarity matrix in terms of items or users
        data (pd.DataFrame): The similarity matrix data in DataFrame format.
        measure (SimilarityMeasure): The SimilarityMeasure object that was used.
    """

    def __init__(
        self, name: str, description: str, data: pd.DataFrame, measure: SimilarityMeasure
    ) -> None:
        super().__init__(name=name, description=description, data=data)
        self._measure = measure

    @property
    def method(self) -> str:
        return self._method.name

    def get_similarity(self, u: int, v: int) -> float:
        if u >= v:  # Swap the values as similarity measures are calculated in u,v order.
            w = u
            u = v
            v = w
        return self._dataframe[(self._dataframe["u"] == u) & (self._dataframe["v"] == v)]["score"]
