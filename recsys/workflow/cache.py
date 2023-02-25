#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/cache.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 07:11:13 am                                             #
# Modified   : Saturday February 25th 2023 09:15:29 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cache Module"""
from datetime import datetime
from dataclasses import dataclass
from typing import Any
import functools
import logging

import isodate
from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.database.cache import CacheDB


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


# ------------------------------------------------------------------------------------------------ #
@inject
def cache(cache_db: CacheDB = Provide[Recsys.db.cache]):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            module = func.__module__
            classname = func.__qualname__
            params = func.__repr__()
            key = module + "_" + classname + "_" + params

            logger = logging.getLogger(f"{module}.{classname}")

            if cache_db.exists(key):
                logger.info(f"Operator {module}.{classname} retrieved data from cache.")
                cache = cache_db.select(key)
                return cache.content

            else:
                try:
                    result = func(self, *args, **kwargs)

                    cache = Cache(key=key, duration=cache_db.duration, content=result)
                    cache_db.insert(cache)

                    return result
                except Exception as e:
                    logger.exception(
                        f"Exception raised in {module}.{classname}. Exception: {str(e)}"
                    )
                    raise e

        return wrapper

    return decorator
