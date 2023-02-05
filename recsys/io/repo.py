#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/repo.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 10:40:29 pm                                                #
# Modified   : Saturday February 4th 2023 10:33:52 pm                                              #
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
import shutil


# ------------------------------------------------------------------------------------------------ #
class Repo:
    """Object repostirory class"""

    def __init__(self, locations: dict = None) -> None:
        # Obtain mode from the environment variable
        load_dotenv()
        mode = os.environ.get("MODE")
        # locations dict is keyed by mode and value is a location
        self._location = locations[mode]
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

    def exists(self, name: str) -> bool:
        """Returns true if the object named exists, False otherwise"""
        with shelve.open(self._location) as db:
            return db.get(name, None) is not None

    def reset(self) -> None:
        """Purges the repo and deletes the shelf"""
        shutil.rmtree(os.path.dirname(self._location), ignore_errors=True)
        os.makedirs(os.path.dirname(self._location), exist_ok=True)
