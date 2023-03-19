#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/event.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 08:04:10 pm                                                #
# Modified   : Saturday March 18th 2023 08:05:16 pm                                                #
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
