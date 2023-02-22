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
# Created    : Sunday January 29th 2023 07:02:57 am                                                #
# Modified   : Wednesday February 22nd 2023 11:24:02 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for entity package."""
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import logging

from recsys.dal.base import DTO


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET ABSTRACT BASE CLASS                                     #
# ------------------------------------------------------------------------------------------------ #
class Data(ABC):
    """Data base class.

    Args:
        name (str): Name of the data object
        description (str): Describes the contents of the data object
        stage (str): The stage within the data flow or lifecycle.

    """

    def __init__(self, name: str, description: str, stage: str) -> None:
        self._name = name
        self._description = description
        self._stage = stage
        self._type = self.__class__.__name__
        self._id = None
        self._filepath = None
        self._profile = None
        self._workspace = self._get_workspace()
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __len__(self) -> int:
        return len(self._data)

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def size(self) -> int:
        return self._size

    @property
    def matrix_size(self) -> int:
        return self._matrix_size

    @property
    def sparsity(self) -> int:
        return self._sparsity

    @property
    def density(self) -> int:
        return self._density

    @property
    def memory_mb(self) -> int:
        return self._memory_mb

    @property
    def cost(self) -> str:
        return self._cost

    @cost.setter
    def cost(self, cost: str) -> None:
        self._cost = cost

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath: str) -> None:
        self._filepath = filepath

    @abstractmethod
    def summarize(self) -> None:
        """Returns a summary of the data"""

    @abstractmethod
    def as_dto(self) -> DTO:
        """Returns a Data Transfer Object representation of the entity."""

    def _get_workspace(self) -> str:
        """Reads the current workspace from the environment variable."""
        load_dotenv()
        return os.getenv("WORKSPACE", "dev")
