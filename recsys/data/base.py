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
# Modified   : Sunday February 19th 2023 04:37:46 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import logging
from typing import Any

from dependency_injector.wiring import Provide, inject

from recsys.data.fio import IOService
from recsys.container import Recsys


# ------------------------------------------------------------------------------------------------ #
class FMS(ABC):
    """Abstract file management class."""

    @inject
    def __init__(self, config: dict, io: IOService = Provide[Recsys.data.io]) -> None:
        self._config = config
        self._io = io()

    def read(self, filepath: str) -> Any:
        """Read the file"""
        return self._io.read(filepath)

    def write(self, data: Any, filepath: str) -> Any:
        """Write data to file"""
        self._io.write(data=data, filepath=filepath)

    @abstractmethod
    def get_filepath(self, name: str, stage: str) -> str:
        """Returns the filepath for the provided name, stage and current environment."""


# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Abstract base class for rdbms databases"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def connect(self) -> None:
        """Connects to the underlying database."""

    @abstractmethod
    def close(self) -> None:
        """Closes connection to underlying database."""

    @abstractmethod
    def command(self, sql: str, args: tuple = None) -> Any:
        """Executes an sql command on the database and returns a cursor object."""

    @abstractmethod
    def insert(self, sql: str, args: tuple = None) -> int:
        """Inserts a row into a table in the database."""

    @abstractmethod
    def select(self, sql: str, args: tuple = None) -> tuple:
        """Returns a single row from the database"""

    @abstractmethod
    def select_all(self, sql: str, args: tuple = None) -> tuple:
        """Returns multiple rows from the database"""

    @abstractmethod
    def update(self, sql: str, args: tuple = None) -> None:
        """Performs an update on existing data in the database."""

    @abstractmethod
    def delete(self, sql: str, args: tuple = None) -> None:
        """Deletes existing data from a database table."""
