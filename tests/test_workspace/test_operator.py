#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /tests/test_workspace/test_operator.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday March 5th 2023 12:33:42 am                                                   #
# Modified   : Sunday March 5th 2023 12:38:52 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from datetime import datetime

from recsys.operator.base import Operator


# ------------------------------------------------------------------------------------------------ #
#                                      TEST OPERATOR                                               #
# ------------------------------------------------------------------------------------------------ #
class TestOperator(Operator):
    """Operator does nothing."""

    def __init__(self) -> None:
        super().__init__()

    def execute(self, *args, **kwargs) -> None:
        """Downloads a zipfile."""
        self._logger.debug(datetime.now())
