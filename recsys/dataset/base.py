#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 03:40:09 pm                                              #
# Modified   : Sunday March 5th 2023 10:17:34 pm                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for domain package."""
from abc import ABC, abstractmethod
from copy import deepcopy
import logging

import numpy as np
import pandas as pd

from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET CLASS                                                   #
# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):  # pragma: no cover
    """Base class for tabular datasets.
    Args:
        name (str): Name of the dataset
        description (str): Describes the contents of the dataset
        data (pd.DataFrame): The pandas dataframe
        datasource (str): The original source of the data
    """

    def __init__(
        self,
        name: str,
        description: str,
        data: pd.DataFrame,
        datasource: str = "movielens25m",
    ) -> None:

        self._name = name
        self._description = description
        self._data = data
        self._datasource = datasource
        self._summary = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def name(self) -> np.array:
        return self._name

    @property
    def description(self) -> np.array:
        return self._description

    @property
    def columns(self) -> np.array:
        return self._data.columns

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def size(self) -> int:
        return self._size

    @property
    def datasource(self) -> str:
        return self._datasource

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._data.head(n)

    def to_df(self) -> pd.DataFrame:
        return deepcopy(self._data)

    def to_parquet(self, filepath: str) -> None:
        IOService.write(filepath=filepath, data=self._data)

    def to_csv(self, filepath: str) -> None:
        IOService.write(filepath=filepath, data=self._data)

    def describe(self) -> pd.DataFrame:
        """DataFrame containing summary and statistical information"""
        return self._data.describe().T

    def summary(self) -> None:
        self._summarize()
        return self._summary

    @abstractmethod
    def _summarize(self) -> None:
        """Computes summary statistics."""
