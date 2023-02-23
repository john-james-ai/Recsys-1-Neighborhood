#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/core/job.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:57:31 pm                                            #
# Modified   : Wednesday February 22nd 2023 09:16:28 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Job  and Task Module"""
import logging
from abc import ABC, abstractmethod
from recsys.dal.base import Repo


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for Tasks"""

    def __init__(self, name: str, description: str, category: str = None, *args, **kwargs) -> None:
        self._id = None
        self._name = name
        self._description = description
        self._category = category
        self._status = "pending"
        self._exception = False
        self._inspec = None  # Input specification
        self._output = None  # Output specification
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    from workspaces import Repo

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id: int) -> int:
        self._id = id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def set_inspec(self, *args, **kwargs) -> None:
        """Sets the input specifications"""

    @abstractmethod
    def set_outspec(self, *args, **kwargs) -> None:
        """Sets the output specifications"""

    @abstractmethod
    def repo(self) -> str:
        """A repository instance"""

    @repo.setter
    def repo(self, repo: Repo) -> Repo:
        """Sets the repository"""

    @property
    def category(self) -> str:
        return self._category

    @property
    def status(self) -> str:
        return self._status

    @property
    def exception(self) -> str:
        return self._exception

    @property
    @abstractmethod
    def run(self) -> None:
        """Executes the task."""

    def setup(self, *args, **kwargs) -> None:
        """Prerun setup"""

    def teardown(self, *args, **kwargs) -> None:
        """Post-run teardown"""


# ------------------------------------------------------------------------------------------------ #
class Job(ABC):
    """Abstract base class for Job"""

    def __init__(self, name: str, description: str, category: str = None, *args, **kwargs) -> None:
        self._name = name
        self._description = description
        self._category = category
        self._status = "pending"
        self._exception = False
        self._inspec = None  # Input specification
        self._output = None  # Output specification
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Add Task to the Job"""

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id: int) -> int:
        self._id = id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def set_inspec(self, *args, **kwargs) -> None:
        """Sets the input specifications"""

    @abstractmethod
    def set_outspec(self, *args, **kwargs) -> None:
        """Sets the output specifications"""

    @abstractmethod
    def repo(self) -> str:
        """A repository instance"""

    @repo.setter
    def repo(self, repo: Repo) -> Repo:
        """Sets the repository"""

    @property
    def category(self) -> str:
        return self._category

    @property
    def status(self) -> str:
        return self._status

    @property
    def exception(self) -> str:
        return self._exception

    @property
    @abstractmethod
    def run(self) -> None:
        """Executes the task."""

    def setup(self, *args, **kwargs) -> None:
        """Prerun setup"""

    def teardown(self, *args, **kwargs) -> None:
        """Post-run teardown"""
