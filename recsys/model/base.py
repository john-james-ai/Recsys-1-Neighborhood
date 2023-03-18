#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/model/base.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 07:02:43 am                                                  #
# Modified   : Friday March 17th 2023 09:54:52 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from recsys.data.dataset import Dataset


# ------------------------------------------------------------------------------------------------ #
class Model(ABC):
    """Abstract base class for model objects."""

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """Sets dataset"""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes prediction"""

    @abstractmethod
    def score(self, X: pd.DataFrame, y: np.array) -> float:
        """Scores the predictions"""
