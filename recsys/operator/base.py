#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:41:57 pm                                             #
# Modified   : Friday February 24th 2023 11:35:43 pm                                               #
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

from recsys.persistence.fio import IOService
from recsys.container import Recsys
from recsys.workflow.event import event_log


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Base class for all operators"""

    @inject
    def __init__(
        self,
        source: str,
        destination: str,
        force: bool = False,
        fio: IOService = Provide[Recsys.services.fio],
    ) -> None:
        self._source = source
        self._destination = destination
        self._force = force
        self._started = None
        self._ended = None
        self._duration = None
        self._fio = fio
        self._status = "created"
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def status(self) -> str:
        return self._status

    @property
    def source(self) -> str:
        return self._source

    @property
    def destination(self) -> str:
        return self._destination

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

    @event_log
    def execute(self, data: Any = None) -> Union[Any, None]:
        """Performs the operation."""
        self._setup()
        if self._skip():
            pass
        else:
            data = self.run(data)
        self._teardown()
        return data

    @abstractmethod
    def run(self, data: Any = None) -> Union[Any, None]:
        """Performs the operation"""

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
