#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/corating.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 11:55:43 am                                                #
# Modified   : Thursday March 2nd 2023 08:26:23 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Corating Data Module"""

import numpy as np
import pandas as pd

from recsys.dataset.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class Corating(Dataset):
    """DataFrame listing corated items or users.

    Corated items can be a list of items rated by a pair of users, or a list of users who have
    rated a pair of items. The specific implementation is determined by the keyvar, and coratevar
    variables.
    Args:
        name (str): The name of the corating dataset
        description (str): Description indicating whether this is a user or item corating index.
        data (str): DataFrame containing two columns: a column containing tuples of users or items
            and a column containing the items or users rating, or rated by the tuple pair
        keyvar (tuple): A pair of users or items. Valid values are ['userId', 'movieId']
        coratevar  (int): If keyvar is a pair of users, the coratevar is the column containing
            the items rated by the pair of users. If the keyvar is a pair of items, the coratevar
            is the variable containing the users that have rated both items. Valid values are
            ['userId', 'movieId']
    """

    def __init__(
        self,
        name: str,
        description: str,
        data: pd.DataFrame,
        keyvar: str = "userId",
        coratevar: str = "movieId",
    ) -> None:
        super().__init__(name=name, description=description, data=data)
        self._keyvar = keyvar
        self._coratevar = coratevar
        self._summary = None
        self._counts = None

    def get_corating(self, key: tuple) -> np.array:
        """Provides corated items or users"""
        return self._data[self._data[self._keyvar] == key][self._coratevar].values

    def get_counts(self) -> pd.DataFrame:
        if self._counts is None:
            self._counts = self._data.value_counts()
        return self._counts

    def _summarize(self) -> None:
        self._summary = {}
        keyvar = self._keyvar + "_pairs"
        coratevar = self._coratevar + "s"
        if self._summary is None:
            if self._counts is None:
                self._counts = self._data.value_counts()
            self._summary[keyvar] = self._data[self._keyvar].nunique()
            self._summary[coratevar] = self._data[self._coratevar].describe().T
        print(f"\nThere are {self._summary[keyvar]} unique pairs")
        print(f"Summary Statistics of Counts per Pair\n{self._summary[coratevar]}")
