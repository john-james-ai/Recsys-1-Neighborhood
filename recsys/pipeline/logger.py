#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/pipeline/logger.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:53:01 pm                                             #
# Modified   : Saturday February 18th 2023 11:37:07 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Logging module for workflow package"""
import functools
from datetime import datetime
import logging


# ------------------------------------------------------------------------------------------------ #
#                                OPERATOR LOG DECORATOR                                            #
# ------------------------------------------------------------------------------------------------ #
def logger(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        module = func.__module__
        classname = func.__qualname__
        name = self.__dict__.get("_name", None)
        description = self.__dict__.get("_description", None)

        logger = logging.getLogger(f"{module}.{classname}")

        try:
            logger.info(f"Started {module}.{classname} instance name: {name} {description}")

            started = datetime.now()
            result = func(self, *args, **kwargs)
            ended = datetime.now()
            duration = round((ended - started).total_seconds(), 3)

            logger.info(
                f"Completed {module}.{classname} instance name: {name}. Duration: {duration}"
            )
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
            raise e

    return wrapper
