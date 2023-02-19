#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/pipeline/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:41:57 pm                                             #
# Modified   : Sunday February 19th 2023 04:28:36 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.pipeline.logger import logger


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Base class for workflow operators"""

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @logger
    def run(self, *args, **kwargs) -> Union[pd.DataFrame, None]:
        """Performs the operation."""


# ------------------------------------------------------------------------------------------------ #
class Pipeline(ABC):
    """Base class for Pipeline objects.

    Args:
        name (str): name of the pipeline
        description (str): Description of what the pipeline does.
        io (IOService): Input/output service

    """

    @inject
    def __init__(self, name: str, description: str, io: IOService = Provide[Recsys.io.io]) -> None:
        self._name = name
        self._description = description
        self._io = io()
        self._tasks = {}

    @abstractmethod
    def add_task(self, operator: Operator) -> None:
        """Add a task performed by a parameterized operator to the pipeline"""

    @logger
    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline"""
