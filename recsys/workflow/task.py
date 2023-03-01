#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/task.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 06:29:22 pm                                             #
# Modified   : Tuesday February 28th 2023 11:51:16 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Any

from recsys.workflow.base import Event
from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
class Task(Event):
    def __init__(self, name: str, description: str, operator: Operator, *args, **kwargs) -> None:
        super().__init__(name=name, description=description)
        self._operator = operator

    def run(self, data: Any = None) -> Any:
        """Executes the task"""
        self._setup()
        data = self._operator.execute(data)
        self._teardown()
        return data


# ------------------------------------------------------------------------------------------------ #
