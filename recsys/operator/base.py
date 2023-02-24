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
# Modified   : Thursday February 23rd 2023 01:47:26 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Base class for all operators"""

    __name = "base_operator"
    __description = "Abstract Base Class for Operators"
    __category = "Base Class"

    def __init__(self, force: bool = False, *args, **kwargs) -> None:
        self._bool = bool

    def run(self, *args, **kwargs) -> None:
        """Performs the operation."""

    @abstractmethod
    def skipif(self) -> bool:
        """Returns True if force is False and task endpoint already exists."""
