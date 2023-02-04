#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/matrix.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 08:37:49 pm                                                #
# Modified   : Friday February 3rd 2023 10:46:07 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Utility Matrix Module"""
import pandas as pd
import numpy as np


from recsys.neighborhood.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(Matrix):
    def __init__(self, name: str, data: pd.DataFrame) -> None:
        super().__init__(name=name, data=data)

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.shape[0] * self._data.shape[1]

    @property
    def memory(self) -> int:
        return self._data.memory_usage(deep=True).sum()

    def load(self) -> None:
        self._data = self._repo.get(self._name)

    def save(self) -> None:
        if self._repo.exists(self._name):
            self._repo.update(name=self._name, item=self._data)
        else:
            self._repo.add(name=self._name, item=self._data)


# ------------------------------------------------------------------------------------------------ #
class UserSimilarityMatrix(SimilarityMatrix):
    def get_similarity(self, a: int, b: int) -> float:
        u, v = np.sort([a, b])
        return self._data[(self._data["u"] == u) & (self._data["v"] == v)].values[0]


# ------------------------------------------------------------------------------------------------ #
class ItemSimilarityMatrix(SimilarityMatrix):
    def get_similarity(self, a: int, b: int) -> float:
        i, j = np.sort([a, b])
        return self._data[(self._data["i"] == i) & (self._data["j"] == j)].values[0]
