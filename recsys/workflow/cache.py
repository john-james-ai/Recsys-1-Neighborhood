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
# Created    : Saturday February 25th 2023 11:52:22 pm                                             #
# Modified   : Sunday February 26th 2023 05:41:04 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cache and Decorator Module"""
import sys
import logging


from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.system.cache import Cache

# ------------------------------------------------------------------------------------------------ #


def cache(func):
    @inject
    def wrapper(*args, cache_db=Provide[Recsys.cache], **kwargs):

        module = func.__module__
        classname = func.__qualname__

        logger = logging.getLogger(f"{module}.{classname}")
        logger.debug(f"\nInside cache for operator {module}.{classname}.")

        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        logger.debug(f"The cache key: {signature}.")

        if cache_db.exists(signature):
            logger.info(f"Operator {module}.{classname} retrieved data from cache.")
            cache = cache_db.select(signature)
            return cache.content
        else:
            logger.debug("Prior results not found in cache.")
            try:
                result = func(*args, **kwargs)

                cache = Cache(key=signature, duration=cache_db.duration, content=result)
                cache_db.insert(cache)

                logger.debug(f"Wrote {sys.getsizeof(result)} bytes to cache.")

                return result
            except Exception as e:
                logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
                raise e

    return wrapper
