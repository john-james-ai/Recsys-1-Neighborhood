#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 09:34:32 pm                                                 #
# Modified   : Sunday March 5th 2023 12:39:59 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Any, Union
from datetime import datetime
import logging


# ------------------------------------------------------------------------------------------------ #
class Event(ABC):  # pragma: no cover
    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the task"""

    @property
    @abstractmethod
    def description(self) -> str:
        """Returns the description of the task"""

    @property
    @abstractmethod
    def started(self) -> datetime:
        """Returns the datetime the task started."""

    @property
    @abstractmethod
    def ended(self) -> datetime:
        """Returns the datetime the task ended."""

    @property
    @abstractmethod
    def duration(self) -> datetime:
        """Returns the duration of the task."""

    @abstractmethod
    def _setup(self) -> None:
        """Performs required initialization steps before running the task"""

    @abstractmethod
    def _teardown(self) -> None:
        """Wrap up activities."""

    @abstractmethod
    def run(self) -> Union[None, Any]:
        """Runs the task."""


# ------------------------------------------------------------------------------------------------ #
class Builder(ABC):  # pragma: no cover
    """Constructs Configuration file based Pipeline objects"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def pipeline(self):
        """Returns the constructed Pipeline object."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the Pipeline object."""

    @abstractmethod
    def build(self, config_filepath: str) -> None:
        """Constructs a Pipeline object.
        Args:
            config_filepath (str): Pipeline configuration
        """
