#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/assets/data.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 04:43:52 am                                                #
# Modified   : Wednesday March 1st 2023 04:52:10 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import abstractmethod
from typing import Any

import pandas as pd
from recsys.assets.base import Asset


# ------------------------------------------------------------------------------------------------ #
#                                  DATA ASSET CLASS                                                #
# ------------------------------------------------------------------------------------------------ #
class DataAsset(Asset):  # pragma: no cover
    """Base class for data assets.
    Args:
        name (str): Name of the dataset
        description (str): Describes the contents of the dataset
        data (Any): The data content
        datasource (str): The original source of the data
    """

    def __init__(
        self,
        name: str,
        description: str,
        data: Any,
        datasource: str = "movielens25m",
    ) -> None:
        super().__init__(name=name, description=description)
        self._datasource = datasource
        self._data = data

    @property
    def datasource(self) -> str:
        return self._datasource

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Provides summary and descriptive statistics about the asset."""
