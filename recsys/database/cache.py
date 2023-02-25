#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/database/cache.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 08:13:00 am                                             #
# Modified   : Saturday February 25th 2023 09:06:48 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Object Database Module"""
import shelve
from typing import Any
from datetime import datetime

from recsys.database.base import Database
from recsys.workflow.cache import Cache


# ------------------------------------------------------------------------------------------------ #
class CacheDB(Database):
    """Cache object database"""

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self._location = self._config.location
        self._duration = self._config.duration

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

    def connect(self) -> None:
        """Connects to the database."""
        self._connection = shelve.open(self._location)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()
        self._connection = None

    def insert(self, cache: Cache) -> None:
        """Inserts data into the database."""
        if not self.exists(cache.key):
            self._connection[cache.key] = cache
        else:
            msg = f"Object with key {cache.key} already exists in the database."
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

    def update(self, cache: Cache) -> None:
        """Updates data in the database."""
        if self.exists(cache.key):
            self._connection[cache.key] = cache
        else:
            msg = f"Object with key {cache.key} not found in database."
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

    def clean(self) -> None:
        """Removes expired cache."""
        self.connect()
        for key, cache in self._connection.items():
            if cache.expires < datetime.now():
                del self._connection[key]
