#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/system/idgen.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 06:03:54 am                                               #
# Modified   : Sunday February 26th 2023 06:23:46 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Id Generator Module"""

# ------------------------------------------------------------------------------------------------ #
class IDGen:
    def __init__(self, config: dict, db: Database) -> None:
        self._config = config
        self._min = config.get("min")
        self._max = config.get("max")
        self._length = config.get("length")
        self._method = config.get("method")
        self._db = config.get('persistence')

    def get_id(self,)
