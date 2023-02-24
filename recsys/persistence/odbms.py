#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/odbms.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 23rd 2023 04:31:27 am                                             #
# Modified   : Thursday February 23rd 2023 04:48:55 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import shelve

from recsys.persistence.base import Database
from recsys.core.base import Entity


# ------------------------------------------------------------------------------------------------ #
#                                     OBJECT DATABASE                                              #
# ------------------------------------------------------------------------------------------------ #
class ODB(Database):
    """Object database class built on Shelf"""

    __location = "labs/recsys.odb"

    def __init__(self, *args, **kwargs) -> None:
        self._connetion = None
        self._logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    def connect(self) -> None:
        """Connects to the database."""
        self._connection = shelve.open(ODB.__location)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()

    def insert(self, entity: Entity) -> None:
        """Inserts data into a table and returns the last row id."""
        self._connection[entity.name] = entity

    def select(self, name: str) -> Entity:
        """Retrieves an item from object storage.."""
        return self._connection[name]

    def update(self, entity: Entity) -> None:
        """Performs an update on existing data in the database."""
        if not self._connection.get(entity.name, None):
            msg = f"Unable to update item named {entity.name}. It does not exist."
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            self._connection[entity.name] = entity

    def delete(self, name: str) -> None:
        """Deletes existing data."""
        if not self._connection.get(name, None):
            msg = f"Unable to delete item named {name}. It does not exist."
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            del self._connection[name]

    def exists(self, name: str) -> None:
        """Checks existence of an item in the database."""
        return self._connection.get(name, None)
