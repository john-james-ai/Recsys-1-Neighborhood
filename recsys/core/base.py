#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Vedion    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/core/base.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 07:02:57 am                                                #
# Modified   : Wednesday February 22nd 2023 11:23:31 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for entity package."""
from abc import ABC, abstractmethod
import logging
import pandas as pd


# ------------------------------------------------------------------------------------------------ #
#                                  ENTITY ABSTRACT BASE CLASS                                      #
# ------------------------------------------------------------------------------------------------ #
class Entity(ABC):
    """Definess the interface for the project Entity classes.

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
        self._lab = self._get_lab()
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

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
    def lab(self) -> str:
        return self._lab

    @abstractmethod
    def as_dto(self) -> pd.DataFrame:
        """Returns a Data Transfer Object representation of the entity."""
