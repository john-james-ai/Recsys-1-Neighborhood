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
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 02:13:37 pm                                             #
# Modified   : Monday February 20th 2023 10:12:37 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import logging
from typing import Any


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
    def command(self) -> None:
        """Executes the SQL command on the underlying database connection."""

    @abstractmethod
    def insert(self, sql: str, args: tuple = None) -> int:
        """Inserts data into a table and returns the last row id."""

    @abstractmethod
    def select(self, sql: str, args: tuple = None) -> tuple:
        """Performs a select command returning a single instance or row."""

    @abstractmethod
    def select_all(self, sql: str, args: tuple = None) -> list:
        """Performs a select command returning multiple instances or rows."""

    @abstractmethod
    def update(self, sql: str, args: tuple = None) -> None:
        """Performs an update on existing data in the database."""

    @abstractmethod
    def delete(self, sql: str, args: tuple = None) -> None:
        """Deletes existing data."""

    @abstractmethod
    def exists(self, sql: str, args: tuple = None) -> None:
        """Checks existence of an item in the database."""


# ------------------------------------------------------------------------------------------------ #
#                                  FILE MANAGEMENT SYSTEM BASE CLASS                               #
# ------------------------------------------------------------------------------------------------ #
class FMSBase(ABC):
    """Abstract file management service class."""

    @abstractmethod
    def read(self, filepath: str) -> Any:
        """Read the file"""

    @abstractmethod
    def write(self, data: Any, filepath: str) -> Any:
        """Write data to file"""

    @abstractmethod
    def delete(self, filepath: str) -> Any:
        """Write data to file"""

    @abstractmethod
    def get_filepath(self, name: str, stage: str) -> str:
        """Returns the filepath for the provided name, stage and current environment."""
