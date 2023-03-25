#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 20th 2023 10:28:58 pm                                                  #
# Modified   : Monday March 20th 2023 10:29:47 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base DataPrep Module: Defines the Operator Base Class"""
import os
from abc import abstractmethod
from typing import Union

import pandas as pd

from recsys.asset.base import Asset


# ------------------------------------------------------------------------------------------------ #
class Operator(Asset):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow

    Note: Subclasses must define two class members for identification and persistence purposes:
        __name
        __description

    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @classmethod
    def name(cls) -> str:
        return cls.__name

    @property
    @classmethod
    def desc(cls) -> str:
        return cls.__desc

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
