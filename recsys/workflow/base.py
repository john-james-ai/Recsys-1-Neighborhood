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
# Created    : Saturday February 25th 2023 06:29:22 pm                                             #
# Modified   : Saturday February 25th 2023 10:09:17 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Union
import logging


# ------------------------------------------------------------------------------------------------ #
class Event(ABC):
    def __init__(self, name: str, description: str, *args, **kwargs) -> None:
        self._name = name
        self._description = description

        self._started = None
        self._ended = None
        self._duration = 0
        self._status = "created"
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def name(self) -> int:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def status(self) -> str:
        return self._status

    @property
    def started(self) -> datetime:
        return self._started

    @property
    def ended(self) -> datetime:
        return self._ended

    @property
    def duration(self) -> int:
        return self._duration

    def _setup(self) -> None:
        self._started = datetime.now()
        self._status = "started"

    def _teardown(self) -> None:
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        self._status = "success" if not self._status == "exception" else self._status

    @abstractmethod
    def run(self, data: Any = None) -> Union[Any, None]:
        """Fulfills the event"""
