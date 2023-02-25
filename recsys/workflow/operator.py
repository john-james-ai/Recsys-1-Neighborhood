#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/operator.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 05:43:46 am                                             #
# Modified   : Saturday February 25th 2023 09:27:14 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import os
import logging
from typing import Any, Union
from datetime import datetime

from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.io.service import IOService


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow"""

    @inject
    def __init__(self, *args, **kwargs) -> None:
        self._started = None
        self._ended = None
        self._duration = None
        self._status = "created"
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the object."""

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

    def _teardown(self, status: str = "success") -> None:
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        self._status = "success" if self._status == "started" else self._status

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Executes the operation"""


# ------------------------------------------------------------------------------------------------ #
class FileOperator(Operator):
    """Base class for operators that manipulate files."""

    @inject
    def __init__(
        self,
        source: str,
        destination: str,
        force: bool = False,
        fio: IOService = Provide[Recsys.services.fio],
    ) -> None:
        super().__init__()
        self._source = source
        self._destination = destination
        self._force = force
        self._fio = fio

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the object."""

    @property
    def source(self) -> str:
        return self._source

    @property
    def destination(self) -> str:
        return self._destination

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Performs the operation."""

    def _skip(self) -> bool:
        """Used to evaluate whether the operation should be skipped."""
        if self._force:
            return False
        elif os.path.isfile(self._destination) and os.path.exists(self._destination):
            self._logger.info(
                f"{self.__class__.__name__} skipped. Destination file {self._destination} already exists. To overwrite set force to True."
            )
            self._status = "skipped"
            return True
        elif (
            os.path.isdir(self._destination)
            and os.path.exists(self._destination)
            and len(os.listdir(self._destination)) > 0
        ):
            self._logger.info(
                f"{self.__class__.__name__} skipped. Destination {self._destination} is not empty. To overwrite set force to True."
            )
            self._status = "skipped"
            return True
        else:
            return False


# ------------------------------------------------------------------------------------------------ #
class DataOperator(Operator):
    """Base class for operators that manipulate data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the object."""

    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Performs the operation on data and returns the data"""
