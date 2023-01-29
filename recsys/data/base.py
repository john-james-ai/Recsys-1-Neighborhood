#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/base.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 07:02:57 am                                                #
# Modified   : Sunday January 29th 2023 07:43:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for data package."""
import os
import logging
from abc import ABC, abstractmethod
import pandas as pd

from recsys.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Base class for datasets"""

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._data = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def load(self) -> None:
        self._logger.debug(f"Loading {os.path.basename(self._filepath)}")
        self._data = IOService.read(self._filepath)
        self._logger.debug(f"Load {os.path.basename(self._filepath)} complete.")

    def save(self, filepath: str, force: bool = False) -> None:

        if self._force or not os.path.exists(filepath):
            self._logger.debug(f"Saving {os.path.basename(filepath)}")
            IOService.write(filepath=filepath, data=self._data)
            self._logger.debug(f"Save {os.path.basename(filepath)} complete.")
        else:
            self._logger.debug(f"Saving {os.path.basename(filepath)} skipped. File already exists.")

    @abstractmethod
    def summarize(self) -> pd.dataFrame:
        """Provides summary of entity"""

    @abstractmethod
    def split(self, train_prop: float, train_filepath: str, test_filepath: str) -> None:
        """Splits dataset into train and test set"""
