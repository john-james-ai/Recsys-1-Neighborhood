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
# Modified   : Tuesday February 28th 2023 05:11:58 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cache Database Module"""
import os
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
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        self._is_connected = False
        self.connect()

    @property
    def filepath(self) -> str:
        return self._filepath

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._is_connected:
            self.close()
        if exc_type is not None:
            self._logger.error(f"\nExecution Type: {exc_type}")
            self._logger.error(f"\nExecution Value: {exc_value}")
            self._logger.error(f"\nTraceback: {traceback}")

    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Connects to the database."""
        self._connection = shelve.open(self._filepath)
        self._is_connected = True

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()
        self._is_connected = False

    def insert(self, key: str, value: Any) -> None:
        """Inserts a key/value pair into the database."""
        self._check_connection()
        if not self.exists(key):
            self._connection[key] = value
            self.close()
        else:
            msg = f"Object with key {key} already exists in the database."
            self._logger.error(msg)
            raise Database.ObjectExistsError(msg)

    def select(self, key: str) -> Any:
        """Retrieves data from the database"""
        self._check_connection()
        try:
            result = self._connection[key]
            self.close()
            return result
        except KeyError:
            msg = f"Object with key {key} not found in database."
            self._logger.error(msg)
            raise Database.ObjectNotFoundError(msg)

    def selectall(self, key: str) -> Any:
        """Retrieves all data from the database"""
        objects = {}
        self._check_connection()
        keys = self._connection.keys()
        if len(keys) == 0:
            msg = f"Database at {self._filename} is empty."
            raise Database.ObjectDBEmpty(msg)
        for key in keys:
            objects[key] = self._connection[key]
        return objects

    def update(self, key: str, value: Any) -> None:
        """Updates an existing object in the database."""
        self._check_connection()
        if self.exists(key):
            self._connection[key] = value
        else:
            msg = f"Object with key {key} not found in database."
            self._logger.error(msg)
            raise Database.ObjectNotFoundError(msg)

    def delete(self, key: str) -> None:
        """Deletes existing data."""
        self._check_connection()
        if self.exists(key):
            del self._connection[key]
        else:
            msg = f"Object with key {key} doesn't exist in the database."
            self._logger.error(msg)
            raise Database.ObjectNotFoundError(msg)

    def exists(self, key: str) -> bool:
        """Checks existence of an item in the database."""
        self._check_connection()
        return key in self._connection.keys()

    def clear(self) -> None:
        """Clears cache of all objects."""
        self._check_connection()
        self._connection.clear()

    def _check_connection(self) -> None:
        if not self._is_connected:
            self.connect()


# ------------------------------------------------------------------------------------------------ #
class CacheDB(ObjectDB):
    """Cache database"""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath=filepath)

    def clean(self) -> None:
        """Removes expired cache."""
        self._check_connection()
        for key, cache in self._connection.items():
            if cache.expires < datetime.now():
                del self._connection[key]
