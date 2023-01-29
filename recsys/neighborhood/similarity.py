#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/similarity.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 07:36:27 am                                                #
# Modified   : Sunday January 29th 2023 08:07:15 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------------------------ #
class Similarity(ABC):
    """Base class for similarity measures"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Computes the similarity between users u, and v"""


# ------------------------------------------------------------------------------------------------ #
class UserSimilarity(Similarity):
    """Base class for user similarity measures"""

    def __init__(self) -> None:
        super().__init__()

    def compute(self, u: pd.DataFrame, v: pd.DataFrame) -> float:
        """Base class for user similarity measures"""


# ------------------------------------------------------------------------------------------------ #
class ItemSimilarity(Similarity):
    """Base class for user similarity measures"""

    def __init__(self) -> None:
        super().__init__()

    def compute(self, i: pd.DataFrame, j: pd.DataFrame) -> float:
        """Base class for user similarity measures"""


# ------------------------------------------------------------------------------------------------ #
class CosignUserSimilarity(UserSimilarity):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, u: pd.DataFrame, v: pd.DataFrame) -> float:
        u_sorted = u.sort_values(by="movieId")
        v_sorted = v.sort_values(by="movieId")
        assert np.equal(u_sorted["movieId"].values == v_sorted["movieId"].values)
        return u_sorted["rating"].values.dot(v_sorted["rating"].values) / np.sqrt(
            np.sum(np.square(u_sorted["rating"].values))
            * np.sum(np.square(v_sorted["rating"].values))
        )
