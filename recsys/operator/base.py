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
# Modified   : Tuesday February 28th 2023 04:13:51 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import logging
from typing import Any, Union


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow"""

    def __init__(self, *args, **kwargs) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Executes the operation"""
