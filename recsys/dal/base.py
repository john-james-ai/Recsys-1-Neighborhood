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
# Created    : Wednesday February 22nd 2023 03:41:26 am                                            #
# Modified   : Wednesday February 22nd 2023 10:58:15 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
#                             DATA TRANSFER OBJECT ABC                                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DTO(ABC):
    """Base definition of a Data Transfer Object."""

    id: int = None
    name: str = None
    type: str = None
    description: str = None
    lab: str = None
    filepath: str = None


# ------------------------------------------------------------------------------------------------ #
#                              DATA ACCESS OBJECT ABC                                              #
# ------------------------------------------------------------------------------------------------ #
class DAOBase(ABC):
    """Defines the DAO interface."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def create(self, dto: DTO) -> int:
        """Writes the DTO to the database and returns the last row id

        Args:
            dto (DTO): Data transfer object.
        """

    @abstractmethod
    def read(self, id: int) -> DTO:
        """Returns a DTO from the database.

        Args:
            id (int): The object id.
        """

    @abstractmethod
    def read_all(self) -> pd.DataFrame:
        """Reads all rows from database and return a dataframe"""

    @abstractmethod
    def update(self, dto: DTO) -> None:
        """Updates a DTO in the database

        Args:
            dto (DTO): Data transfer object."""

    @abstractmethod
    def delete(self, id: int) -> None:
        """Deletes an instance from the database.

        Args:
            id (int): The object id.
        """

    @abstractmethod
    def exists(self, id: int) -> bool:
        """Evaluates existence of an object by id

        Args:
            id (int): The object id.
        """
