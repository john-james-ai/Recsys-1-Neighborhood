#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/datastore/repo.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 08:45:00 am                                               #
# Modified   : Tuesday February 28th 2023 11:30:55 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):  # pragma: no cover
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
