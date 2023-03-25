#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/normalize.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 06:41:14 am                                                #
# Modified   : Monday March 20th 2023 09:49:58 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Prep: Normalize Module"""
from typing import Union
import logging

from pandas import pd

from recsys import Dataset
from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                            MINIMUM ITEMS PER USER                                                #
# ------------------------------------------------------------------------------------------------ #
class NormalizeOperator(Operator):
    """Normalizes ratings values via mean centering.

    Args:
        by (str): The column containing the grouping variable.
        epsilon (float): A factor added to the mean centered ratings to avoid zero values.
            Default = 1e-9

    """

    __name = "normalize_operator"
    __desc = "Mean centers ratings."

    def __init__(
        self,
        by: str,
        rating_col: str = "rating",
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self._by = by
        self._rating_col = rating_col
        self._epsilon = epsilon
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Mean centers ratings.

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """
        try:
            # Obtain average ratings
            rbar = data.groupby(by=self._by)[self._rating_col].mean().reset_index()
            rbar.columns = [self._by, "rbar"]

            # Merge with ratings dataset
            data = data.merge(rbar, on=self._by, how="left")

            # Compute centered rating and drop the average rating column
            data[self._by] = data[self._rating_col] - data["rbar"]
            data = data.drop(columns=["rbar"])
            return data
        except KeyError:
            msg = f"Column {self._by} is not valid."
            self._logger.error(msg)
            raise ValueError(msg)
