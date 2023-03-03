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
# Created    : Tuesday February 28th 2023 04:13:11 pm                                              #
# Modified   : Friday March 3rd 2023 01:55:31 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC, abstractmethod
import logging
from typing import Any, Union

from atelier.persistence.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow"""

    def __init__(
        self, source: str = None, destination: str = None, force: bool = False, *args, **kwargs
    ) -> None:
        self._source = source
        self._destination = destination
        self._force = force
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Executes the operation"""

    def _get_data(self, filepath: str) -> Any:
        try:
            return IOService.read(filepath)
        except Exception as e:
            self._logger.error(e)
            raise

    def _put_data(self, filepath: str, data: Any) -> None:
        if filepath is not None:
            IOService.write(filepath=filepath, data=data)

    def _skip(self, endpoint: str) -> bool:
        """Determines of operation should be skipped if endpoint already exists."""
        if endpoint is None or self._force is True:
            return False
        elif os.path.isfile(endpoint) and not self._force:
            self._logger.info(f"{self.__class__.__name__} skipped. Endpoint already exists.")
            return True
        elif os.path.isdir(endpoint) and len(os.listdir(endpoint)) > 0:
            self._logger.info(f"{self.__class__.__name__} skipped. Endpoint already exists.")
            return True
        else:
            return False
