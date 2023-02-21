#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/entity/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 07:02:57 am                                                #
# Modified   : Monday February 20th 2023 08:31:06 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for domain package."""
from dotenv import load_dotenv
import logging
from abc import ABC, abstractmethod

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET BASE CLASS                                              #
# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Base class for datasets.

    Args:
        name (str): Name of the dataset
        description (str): Describes the contents of the dataset
        workspace (str): The workspace in which the dataset was created.
        stage (str): Either 'source', 'stage', or 'storage'.

    """

    def __init__(
        self, name: str, description: str, workspace: str = "dev", stage: str = "staged"
    ) -> None:
        load_dotenv()
        self._name = name
        self._description = description
        self._workspace = workspace
        self._stage = stage
        self._type = self.__class__.__name__
        self._data = None
        self._filepath = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __len__(self) -> int:
        return self._data

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id

    @property
    def name(self) -> int:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def description(self) -> str:
        return self._description

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def workspace(self) -> str:
        return self._workspace

    @abstractmethod
    def profile(self) -> pd.DataFrame:
        """DataFrame containing summary and statistical information"""
