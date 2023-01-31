#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/persistence.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 10:40:29 pm                                                #
# Modified   : Monday January 30th 2023 11:26:03 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Persistence Module"""
import os
from dotenv import load_dotenv
from typing import Any
import shelve
import logging


# ------------------------------------------------------------------------------------------------ #
class Repo:
    """Object repostirory class"""

    def __init__(self, location: str = None) -> None:
        # Default location is an environment variable
        load_dotenv()
        self._location = location or os.environ.get("ODB")
        # Ensure the location directory exists
        os.makedirs(os.path.dirname(self._location), exist_ok=True)

        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def get(self, name: str) -> Any:
        """Returns an object from the repository."""
        with shelve.open(self._location) as db:
            if db.get(name, None) is None:
                msg = f"Object {name} does not exist."
                self._logger.error(msg)
                raise FileNotFoundError(msg)
            return db[name]

    def add(self, name: str, item: Any) -> None:
        """Adds an object to the repository"""
        with shelve.open(self._location) as db:
            if db.get(name, None) is not None:
                msg = f"Unable to add {name} as it already exists. "
                self._logger.error(msg)
                raise FileExistsError(msg)
            db[name] = item

    def update(self, name: str, item: Any) -> None:
        """Updates an existing item in the repository."""
        with shelve.open(self._location) as db:
            if db.get(name, None) is None:
                msg = f"Unable to update {name} as it does not exist."
                self._logger.error(msg)
                raise FileNotFoundError(msg)
            db[name] = item

    def delete(self, name: str) -> None:
        """Deletes a named item from the repository."""
        with shelve.open(self._location) as db:
            if db.get(name, None) is None:
                msg = f"Unable to delete {name} as it does not exist."
                self._logger.error(msg)
                raise FileNotFoundError(msg)
            del db[name]
