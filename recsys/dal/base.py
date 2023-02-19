#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dal/base.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 06:01:04 pm                                             #
# Modified   : Sunday February 19th 2023 11:49:58 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Any

from recsys.domain.base import Entity

# ------------------------------------------------------------------------------------------------ #
IMMUTABLE_TYPES: tuple = (str, int, float, bool, type(None))
SEQUENCE_TYPES: tuple = (list, tuple)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SQL(ABC):  # pragma: no cover
    """Base class for SQL Command Objects."""


# ------------------------------------------------------------------------------------------------ #
#                              DATA TRANSFER OBJECT ABC                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DTO(ABC):  # pragma: no cover
    """Data Transfer Object"""

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Config object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v.strftime("%H:%M:%S on %m/%d/%Y")
        elif isinstance(v, dict):
            return v
        elif hasattr(v, "as_dict"):
            return v.as_dict()
        else:
            """Else nothing. What do you want?"""


# ================================================================================================ #
#                                          DAO                                                     #
# ================================================================================================ #


class DAO(ABC):
    """Base class for the Data Access Objects."""

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def add(self, dto: DTO) -> None:
        pass

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def get(self, id: int) -> DTO:
        pass

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def get_all(self) -> list:
        pass

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def update(self, dto: DTO) -> int:
        pass

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def delete(self, id: int) -> int:
        pass

    # -------------------------------------------------------------------------------------------- #
    @abstractmethod
    def exists(self, id: int) -> int:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                  REPO BASE CLASS                                                 #
# ------------------------------------------------------------------------------------------------ #
class Repo:
    """Repository base class"""

    def __init__(self, dao: DAO) -> None:
        self._dao = dao()
        # Obtain mode from the environment variable
        load_dotenv()
        mode = os.environ.get("ENV")
        # locations dict is keyed by mode and value is a location
        self._location = locations[mode]
        # Ensure the location directory exists
        os.makedirs(os.path.dirname(self._location), exist_ok=True)
        self._io = io()

        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def get(self, id: int) -> Any:
        """Returns an item from the repository."""

    def add(self, *args, **kwargs) -> None:
        """Adds an item to the repository"""

    def update(
        self,
        *args,
    ) -> None:
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
