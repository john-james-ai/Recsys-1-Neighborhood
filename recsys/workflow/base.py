#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:41:57 pm                                             #
# Modified   : Monday February 20th 2023 09:54:00 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from sqlalchemy import engine
from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.workflow.profile import task_profile


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Base class for workflow operators"""

    def __init__(
        self,
        name: str,
        description: str,
        workspace: str = "dev",
        force: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._name = name
        self._description = description
        self._workspace = workspace
        self._force = force

    @task_profile
    def run(self, *args, **kwargs) -> None:
        """Performs the operation."""

    @abstractmethod
    def skip(self) -> bool:
        """Returns True if force is False and task endpoint already exists."""


# ------------------------------------------------------------------------------------------------ #
class Pipeline(ABC):
    """Base class for Pipeline objects.

    Args:
        name (str): name of the pipeline
        description (str): Description of what the pipeline does.
        io (IOService): Input/output service

    """

    @inject
    def __init__(
        self, name: str, description: str, engine: engine = Provide[Recsys.data.db]
    ) -> None:
        self._name = name
        self._description = description
        self._iengine
        self._tasks = {}

    @abstractmethod
    def add_task(self, operator: Operator) -> None:
        """Add a task performed by a parameterized operator to the pipeline"""

    @logger
    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline"""
