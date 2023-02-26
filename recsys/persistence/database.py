#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/database.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 11:42:47 am                                               #
# Modified   : Sunday February 26th 2023 12:57:18 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Any


# ------------------------------------------------------------------------------------------------ #
#                                       DATABASE                                                  #
# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Abstract Base Class for Database"""

    @abstractmethod
    def connect(self) -> None:
        """Connects to the database."""

    @abstractmethod
    def close(self) -> None:
        """Closes the underlying database connection."""

    @abstractmethod
    def insert(self, key: str, value: Any) -> None:
        """Inserts a key/value pair into the database."""

    @abstractmethod
    def select(self, key: str) -> Any:
        """Retrieves data from the database"""

    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """Updates an existing object in the database."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Deletes existing data."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Checks existence of an item in the database."""
