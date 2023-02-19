#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/domain/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 07:02:57 am                                                #
# Modified   : Sunday February 19th 2023 05:26:11 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for domain package."""
import os
from dotenv import load_dotenv
import logging
from abc import ABC, abstractmethod
from typing import Union, Any

import pandas as pd

from recsys.dal.base import DTO
from recsys.dal.dto import DatasetDTO, ModelDTO
from atelier.utils.memory import get_size


# ------------------------------------------------------------------------------------------------ #
#                                  ENTITY BASE CLASS                                               #
# ------------------------------------------------------------------------------------------------ #
class Entity(ABC):
    """Base class for datasets, models, and other entities with a lifespan.

    Args:
        name (str): Name for the object
        description (str): Description of what the entity contains
        stage (str): Categorization used by subclasses
        env (str): Optional. Either 'dev', 'test', or 'prod'. If None, it defaults to the
            environment value in the environment variable.

    """

    def __init__(
        self,
        name: str,
        description: str,
        stage: str,
        env: str = None,
    ) -> None:
        self._name = name
        self._description = description
        self._stage = stage
        self._env = env or self._get_env()

        self._id = None
        self._memory = None
        self._cost = None

        self._type = self.__class__.__name__

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
    def env(self) -> str:
        return self._env

    @property
    def memory(self) -> float:
        self._memory = get_size(self)
        return self._memory

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, cost: float) -> None:
        self._cost = cost

    @abstractmethod
    def as_dto(self) -> DTO:
        """Returns a Data Transfer Object representation of the entity."""

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """DataFrame containing summary and statistical information"""

    def _get_env(self) -> str:
        """Retrieves tje current environment from the environment variable."""
        load_dotenv()
        return os.getenv("ENV")


# ------------------------------------------------------------------------------------------------ #
#                                  DATASET BASE CLASS                                              #
# ------------------------------------------------------------------------------------------------ #
class Dataset(Entity):
    """Dataset base class.

    Args:
        name (str): Name for the object
        description (str): Description of what the entity contains
        stage (str): Categorization used by subclasses
        env (str): Optional. Either 'dev', 'test', or 'prod'. If None, it defaults to the
            environment value in the environment variable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        data: Union[pd.DataFrame, dict],
        stage: str,
        env: str = None,
    ) -> None:
        super().__init__(name=name, description=description, stage=stage, env=env)
        self._data = data

    @property
    def nrows(self) -> int:
        """Returns number of rows in dataset"""
        return int(self._data.shape[0])

    @property
    def ncols(self) -> int:
        """Returns number of columns in dataset"""
        return int(self._data.shape[1])

    @property
    def size(self) -> int:
        """Returns number cells in the dataframe."""
        return int(self.nrows * self.ncols)

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """Returns a data frame with summary statistics"""

    def as_dto(self) -> DatasetDTO:
        """Returns a Data Transfer Object representation of the entity."""
        return DatasetDTO(
            type=self._type,
            name=self._name,
            description=self._description,
            data=self._data,
            stage=self._stage,
            env=self._env,
            memory=self._memory,
            cost=self._cost,
            size=self._size,
            nrows=self._nrows,
            ncols=self._ncols,
        )


# ------------------------------------------------------------------------------------------------ #
#                                   MODEL BASE CLASS                                               #
# ------------------------------------------------------------------------------------------------ #
class Model(Entity):
    """Model base class.

    Args:
        name (str): Name for the object
        description (str): Description of what the entity contains
        stage (str): Categorization used by subclasses
        env (str): Optional. Either 'dev', 'test', or 'prod'. If None, it defaults to the
            environment value in the environment variable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        dataset: Dataset,
        model: Any,
        stage: str,
        metric: str,
        env: str = None,
        score: str = None,
    ) -> None:
        super().__init__(name=name, description=description, stage=stage, env=env)
        self._dataset = dataset
        self._model = model
        self._metric = metric
        self._score = score

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        self._score = score

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """Returns a model summary statistics"""

    def as_dto(self) -> ModelDTO:
        """Returns a Data Transfer Object representation of the entity."""
        return ModelDTO(
            type=self._type,
            name=self._name,
            description=self._description,
            model=self._model,
            stage=self._stage,
            env=self._env,
            memory=self._memory,
            cost=self._cost,
            size=self._size,
            metric=self._metric,
            score=self._score,
        )
