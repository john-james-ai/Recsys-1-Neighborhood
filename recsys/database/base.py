#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/database/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 08:13:42 am                                             #
# Modified   : Saturday February 25th 2023 08:28:46 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import logging
from typing import Union, Any


# ------------------------------------------------------------------------------------------------ #
#                                           DATABASE                                               #
# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Abstract database class"""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def connect(self) -> None:
        """Connects to the database."""

    @abstractmethod
    def close(self) -> None:
        """Closes the underlying database connection."""

    @abstractmethod
    def insert(self, *args, **kwargs) -> Union[int, None]:
        """Inserts data into the database."""

    @abstractmethod
    def select(self, *args, **kwargs) -> Any:
        """Retrieves data from the database"""

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates data in the database."""

    @abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """Deletes existing data."""

    @abstractmethod
    def exists(self, *args, **kwargs) -> bool:
        """Checks existence of an item in the database."""
