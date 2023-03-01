#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/assets/dataset.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 03:40:09 pm                                              #
# Modified   : Tuesday February 28th 2023 04:08:16 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for domain package."""
from copy import deepcopy

import numpy as np
import pandas as pd

from recsys.persistence.io import IOService
from recsys import Asset


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET BASE CLASS                                              #
# ------------------------------------------------------------------------------------------------ #
class Dataset(Asset):  # pragma: no cover
    """Base class for datasets.
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
        super().__init__(name=name, description=description)
        self._datasource = datasource
        self._data = data

        self._profiled = False
        self._profile = None
        self._run_profile()

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

    def profile(self) -> pd.DataFrame:
        self._run_profile()
        return self._profile

    def _run_profile(self) -> None:
        if not self._profiled:
            self._nrows = self._data.shape[0]
            self._ncols = self._data.shape[1]
            self._size = self._nrows * self._ncols
            self._memory = (
                str(round(self._data.memory_usage(deep=True).sum() / 1024**2, 3)) + " Mb"
            )

            d = {}
            d["id"] = self._id
            d["name"] = self._name
            d["type"] = self._type
            d["description"] = self._description
            d["workspace"] = self._workspace
            d["nrows"] = self._data.shape[0]
            d["ncols"] = self._data.shape[1]
            d["size"] = self._size
            d["memory"] = self.memory

            self._profile = pd.DataFrame.from_dict(data=d, orient="index", columns=["Metric"])
        self._profiled = True
