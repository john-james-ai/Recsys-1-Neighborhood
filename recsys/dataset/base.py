#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 12:41:33 am                                               #
# Modified   : Sunday February 26th 2023 11:50:05 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for domain package."""
import os
from dotenv import load_dotenv
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from recsys.io.service import IOService
from recsys import Asset


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET BASE CLASS                                              #
# ------------------------------------------------------------------------------------------------ #
class Dataset(Asset):
    """Base class for datasets.
    Args:
        name (str): Name of the dataset
        description (str): Describes the contents of the dataset
        workspace (str): The workspace in which the dataset was created.
        stage (str): Either 'source', 'stage', or 'storage'.
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
        self._datasource = datasource
        self._data = data
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

        self._id = None
        self._oid = None

        load_dotenv()
        self._workspace = os.getenv("workspace")

        self._profiled = False
        self._profile = None
        self._run_profile()

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id: str) -> str:
        self._id = id
        self._oid = self.__class__.__name__.lower() + "_" + self._name + "_" + id

    @property
    def oid(self) -> int:
        return self._oid

    @property
    def name(self) -> int:
        return self._name

    @property
    def description(self) -> str:
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
    def memory(self) -> str:
        return self._memory

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def workspace(self) -> str:
        return self._workspace

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
            d["memory"] = self._memory

            self._profile = pd.DataFrame.from_dict(data=d, orient="index", columns=["Metric"])
        self._profiled = True
