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
# Modified   : Monday March 20th 2023 09:50:58 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Dataset Base Module"""
from __future__ import annotations
from abc import abstractmethod

import pandas as pd
import numpy as np

from recsys.asset.base import Asset


# ------------------------------------------------------------------------------------------------ #
class Dataset(Asset):
    """Asset base class for dataset objects"""

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Returns the shape of the Dataset object."""

    @abstractmethod
    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns n rows from the top of the DataFrame"""

    @property
    @abstractmethod
    def columns(self) -> np.array:
        """Returns an array of the column names in the Dataset"""

    @property
    @abstractmethod
    def nrows(self) -> int:
        """Returns the number of rows in the Dataset"""

    @property
    @abstractmethod
    def ncols(self) -> int:
        """Returns the number of columns in the Dataset"""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of elements in the Dataset"""

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """Returns a DataFrame representation of the Dataset object."""

    def summary(self) -> None:
        self._summarize()

    @abstractmethod
    def _summarize(self) -> None:
        """Computes summary statistics."""
