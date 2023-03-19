#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataset/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 08:42:29 pm                                                  #
# Modified   : Saturday March 18th 2023 08:52:16 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Dataset Base Module"""
from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
import logging

import pandas as pd

from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Abstract base class for dataset objects

    Args:
        name (str): Lowercase name of dataset
        description (str): Description of dataset and its contents
        filepath (str): The persistence location for the Dataset object.
        data (pd.DataFrame): Pandas DataFrame containing the data.
    """

    def __init__(self, name: str, description: str, filepath: str, data: pd.DataFrame) -> None:
        self._name = name
        self._description = description
        self._filepath = filepath
        self._data = data
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def filepath(self) -> str:
        return self._filepath

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
