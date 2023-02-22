#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dal/dba.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 02:16:28 pm                                            #
# Modified   : Wednesday February 22nd 2023 02:35:13 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Definition Object Module."""
from abc import ABC, abstractmethod
import logging

from recsys.persistence.base import Database
from recsys.adapter.base import Adapter


# ------------------------------------------------------------------------------------------------ #
#                                    ABSTRACT DBA                                                  #
# ------------------------------------------------------------------------------------------------ #
class AbstractDBA(ABC):
    "Abstract base class for Data Base Administration objects."

    def __init__(self, adapter: Adapter, database: Database) -> None:
        self._adapter = adapter
        self._database = database
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def create(self) -> None:
        """Creates a database, object store or table."""

    @abstractmethod
    def drop(self) -> None:
        """Drops a database, object store or table."""

    @abstractmethod
    def exists(self) -> bool:
        """Returns True if the object exists, False otherwise."""


# ------------------------------------------------------------------------------------------------ #
#                                TABLE AND DATABASE ADMIN                                          #
# ------------------------------------------------------------------------------------------------ #
class DBA(AbstractDBA):
    """Supports basic database table administration.."""

    def __init__(self, adapter: Adapter, database: Database) -> None:
        super().__init__(adapter=adapter, database=database)

    def create(self) -> None:
        """Creates a table."""
        self._database.connect()

        self._database.create(sql=self._adapter.create.sql, args=self._adapter.create.args)
        msg = self._adapter.create.description
        self._logger.info(msg)

        self._database.commit()
        self._database.close()

    def drop(self) -> None:
        """Drops a table."""
        self._database.connect()

        self._database.drop(sql=self._adapter.drop.sql, args=self._adapter.drop.args)
        msg = self._adapter.drop.description
        self._logger.info(msg)

        self._database.commit()
        self._database.close()

    def exists(self) -> None:
        """Checks existence of a database."""
        self._database.connect()

        result = self._database.exists(sql=self._adapter.exists.sql, args=self._adapter.exists.args)
        msg = self._adapter.exists.description
        self._logger.info(msg)

        self._database.commit()

        return result

    def reset(self) -> None:
        self.drop()
        self.create()
