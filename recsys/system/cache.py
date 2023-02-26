#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/system/cache.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 06:22:39 am                                               #
# Modified   : Sunday February 26th 2023 06:23:30 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from datetime import datetime
import isodate
from typing import Any


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Cache:
    key: str
    duration: str
    content: Any
    expires: datetime = None
    created: datetime = datetime.now()

    def __init__(self) -> None:
        self.expires = self.created + isodate.parse_duration(self.duration)
