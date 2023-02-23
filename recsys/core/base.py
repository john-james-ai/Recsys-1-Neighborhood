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
# Modified   : Wednesday February 22nd 2023 06:21:02 pm                                            #
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
    def n_used(self) -> int:
        return self._n_used

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


# ------------------------------------------------------------------------------------------------ #
#                                  DATASOURCE ABSTRACT BASE CLASS                                  #
# ------------------------------------------------------------------------------------------------ #
class DatasourceABC(ABC):
    """Datasource base class.

    Args:
        name (str): Name of the data object
        title (str): Title for the data source
        description (str): Describes the contents of the data object

    """

    def __init__(self, name: str, description: str, stage: str) -> None:
        self._name = name
        self._description = description
        self._type = self.__class__.__name__
        self._id = None
        self._filepath = None
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
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def authors(self) -> str:
        return self._authors

    @authors.setter
    def authors(self, authors: str) -> None:
        self._authors = authors

    @property
    def publisher(self) -> str:
        return self._publisher

    @publisher.setter
    def publisher(self, publisher: str) -> None:
        self._publisher = publisher

    @property
    def published(self) -> str:
        return self._published

    @published.setter
    def published(self, published: str) -> None:
        self._published = published

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    def version(self, version: str) -> None:
        self._version = version

    @property
    def website(self) -> str:
        return self._website

    @website.setter
    def website(self, website) -> None:
        self._website = website

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, url) -> None:
        self._url = url

    @property
    def doi(self) -> str:
        return self._doi

    @doi.setter
    def doi(self, doi) -> None:
        self._doi = doi

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, email) -> None:
        self._email = email
