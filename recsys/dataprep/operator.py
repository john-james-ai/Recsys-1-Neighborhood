#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/operator.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 09:34:32 pm                                                 #
# Modified   : Saturday March 18th 2023 09:03:27 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base DataPrep Module"""
import os
from abc import ABC, abstractmethod
import logging
from typing import Union

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    @abstractmethod
    def __call__(self, data: pd.DataFrame = None) -> Union[pd.DataFrame, None]:
        """Code from subclass that __call__s the operation"""

    def _skip(self, endpoint: str) -> bool:
        """Determines of operation should be skipped if endpoint already exists."""
        if endpoint is None:
            return False
        elif self._force is True:
            return False
        elif os.path.isfile(endpoint):
            self._logger.info(f"{self.__class__.__name__} skipped. Endpoint already exists.")
            return True
        elif os.path.isdir(endpoint) and len(os.listdir(endpoint)) > 0:
            self._logger.info(f"{self.__class__.__name__} skipped. Endpoint already exists.")
            return True
        else:
            return False
