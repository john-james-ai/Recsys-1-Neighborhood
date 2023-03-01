#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/repo/base.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 02:27:55 am                                                #
# Modified   : Wednesday March 1st 2023 03:47:33 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Any
import logging

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):  # pragma: no cover
    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def add(self, **kwargs) -> str:
        """Adds an object to the repository"""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Obtains an object from persistence by key."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Updates an object in storage"""

    @abstractmethod
    def remove(self, key: str) -> None:
        """Removes an object from storage."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Determines existence of an object with the specified key"""

    @abstractmethod
    def info(self) -> pd.DataFrame:
        """Prints the contents of storage."""
