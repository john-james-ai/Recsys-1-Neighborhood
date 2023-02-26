#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/odb.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 08:13:00 am                                             #
# Modified   : Sunday February 26th 2023 05:40:02 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cache Database Module"""
import shelve
from typing import Any
from datetime import datetime

from recsys.persistence.database import Database


# ------------------------------------------------------------------------------------------------ #
#                                       OBJECT DB                                                  #
# ------------------------------------------------------------------------------------------------ #
class ObjectDB(Database):
    """Object Database"""

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self._filepath = filepath
        self._connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._connection is not None:
            self._connection.close()
        if exc_type is not None:
            self._logger.error(f"\nExecution Type: {exc_type}")
            self._logger.error(f"\nExecution Value: {exc_value}")
            self._logger.error(f"\nTraceback: {traceback}")

    def connect(self) -> None:
        """Connects to the database."""
        self._connection = shelve.open(self._filepath)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()
        self._connection = None

    def insert(self, key: str, value: Any) -> None:
        """Inserts a key/value pair into the database."""
        if not self.exists(key):
            self._connection[key] = value
        else:
            msg = f"Object with key {key} already exists in the database."
            self._logger.error(msg)
            raise FileExistsError(msg)

    def select(self, key: str) -> Any:
        """Retrieves data from the database"""
        try:
            return self._connection[key]
        except KeyError as e:
            msg = f"Object with key {key} not found in database."
            self._logger.error(msg)
            raise FileNotFoundError(e)

    def update(self, key: str, value: Any) -> None:
        """Updates an existing object in the database."""
        if self.exists(key):
            self._connection[key] = value
        else:
            msg = f"Object with key {key} not found in database."
            self._logger.error(msg)
            raise FileNotFoundError(msg)

    def delete(self, key: str) -> None:
        """Deletes existing data."""
        if self.exists(key):
            del self._connection[key]
        else:
            msg = f"Object with key {key} doesn't exist in the database."
            self._logger.error(msg)
            raise FileExistsError(msg)

    def exists(self, key: str) -> bool:
        """Checks existence of an item in the database."""
        return key in self._connection.keys()


# ------------------------------------------------------------------------------------------------ #
class CacheDB(ObjectDB):
    """Cache database"""

    def __init__(self, filepath: str, duration: str) -> None:
        super().__init__(filepath=filepath)
        self._duration = duration

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._connection is not None:
            self._connection.close()
        if exc_type is not None:
            self._logger.error(f"\nExecution Type: {exc_type}")
            self._logger.error(f"\nExecution Value: {exc_value}")
            self._logger.error(f"\nTraceback: {traceback}")

    @property
    def duration(self) -> str:
        return self._config.duration

    def exists(self, key: str) -> bool:
        """Checks existence of an item in the database."""
        return key in self._connection.keys()

    def clean(self) -> None:
        """Removes expired cache."""
        self.connect()
        for key, cache in self._connection.items():
            if cache.expires < datetime.now():
                del self._connection[key]
