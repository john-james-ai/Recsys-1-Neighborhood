#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/cache.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 06:22:39 am                                               #
# Modified   : Tuesday February 28th 2023 09:03:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import logging

from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.persistence.odb import CacheDB


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CacheObject:
    key: str
    content: Any
    ttl: int = 7
    expires: datetime = None
    created: datetime = datetime.now()

    def __post_init__(self) -> None:
        self.expires = self.created + timedelta(days=self.ttl)


# ------------------------------------------------------------------------------------------------ #
class Cache:

    __home = "devstudio"
    __cachedb = "cache.db"

    def __init__(self, database: CacheDB, ttl: int = 7) -> None:
        self._database = database
        self._ttl = ttl

    @property
    def ttl(self) -> int:
        return self._ttl

    @ttl.setter
    def ttl(self, ttl: int) -> None:
        self._ttl = ttl

    def clean(self) -> None:
        """Remove expired items."""
        self._database.clean()
        self._get_size()

    def clear(self) -> None:
        """Clear cache of all objects"""
        self._database.clear()
        self._get_size()


# ------------------------------------------------------------------------------------------------ #


def cache(func):
    @inject
    def wrapper(*args, workspace=Provide[Recsys.workspace], **kwargs):

        cache_mgr = workspace.cache

        module = func.__module__
        classname = func.__qualname__
        logger = logging.getLogger(f"{module}.{classname}")

        key = get_hash(args, kwargs)
        logger.debug(f"\n\n{cache_mgr.duration}")

        cache_mgr.connect()

        if cache_mgr.exists(key):
            logger.info(f"Operator {module}.{classname} retrieving data from cache.")
            cache = cache_mgr.select(key)
            cache_mgr.close()
            return cache.content
        else:
            logger.debug("Prior results not found in cache.")
            try:
                result = func(*args, **kwargs)
                cache = CacheObject(key=key, ttl=cache_mgr.ttl, content=result)
                cache_mgr.insert(key=cache.key, value=cache)
                logger.debug(f"Wrote {sys.getsizeof(result)} bytes to cache.")
                cache_mgr.close()
                return result
            except Exception as e:
                logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
                raise e

    return wrapper


def get_hash(args, kwargs):
    args_repr = [repr(a) for a in args]
    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
    signature = ", ".join(args_repr + kwargs_repr)
    return str(hash(signature))
