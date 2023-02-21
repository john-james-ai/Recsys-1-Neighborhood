#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/pipeline/data.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 11:46:34 pm                                             #
# Modified   : Sunday February 19th 2023 11:50:02 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Pipeline Module"""

from recsys.pipeline.base import Pipeline

# ------------------------------------------------------------------------------------------------ #
class DataPipeline(Pipeline):
    """Data Pipeline Class"""

    def __init__(self, name: str, description: str, io: IOService = Provide[Recsys.io.io]) -> None:
        self._name = name
        self._description = description
        self._io = io()
        self._tasks = {}
