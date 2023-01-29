#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/prep.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 03:53:01 am                                                #
# Modified   : Sunday January 29th 2023 10:03:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Prep Module"""
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

from recsys.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Pearson(SimilarityWeight):
    def __init__(self, infilepath: str, outfilepath: str) -> None:
        super().__init__(infilepath=infilepath, outfilepath=outfilepath)
        self._data = IOService.read(self._infilepath)
        self._weights = pd.DataFrame()

    def __call__(self) -> None:
        self._weights = self._data.groupby("userId").agg(self._compute_weights)
        IOService.write(filepath=self._outfilepath, data=self._weights)

    def _compute_weights(self, user_ratings: pd.DataFrame) -> pd.DataFrame:
        neighbor_ratings = self._get_neighbor_ratings(user_ratings["user_id"][0])
        return neighbor_ratings.groupby("userId").agg(self._compute_neighbor_weight, user_ratings)

    def _get_neighbor_ratings(self, userId: int) -> np.array:
        items = self._data[self._data["userId"] == userId]["movieId"].values
        ratings = self._data[self._data["movieId"].isin(items)]
        return ratings[ratings["userId"] != userId]

    def _compute_neighbor_weight(
        self, neighbor_ratings: pd.DataFrame, user_ratings: pd.DataFrame
    ) -> float:
        neighbor_ratings["rating_centered"] = (
            neighbor_ratings["rating"] - neighbor_ratings["rating"].mean()
        )
        user_ratings["rating_centered"] = user_ratings["rating"] - user_ratings["rating"].mean()
        weight = user_ratings["rating_centered"].dot(neighbor_ratings["rating_centered"]) / np.sqrt(
            np.square(user_ratings["rating_centered"]).sum()
            * np.square(neighbor_ratings["rating_centered"]).sum()
        )
        d = {"u": user_ratings["userId"][0], "v": neighbor_ratings["userId"][0], "weight": weight}
        df = pd.DataFrame.from_dict(data=d, orient="columns", index=[user_ratings["userId"][0]])
        return df
