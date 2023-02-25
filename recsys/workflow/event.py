#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/event.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 09:10:34 pm                                               #
# Modified   : Saturday February 25th 2023 02:03:30 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Event Logging Module"""
import functools
from datetime import datetime
import logging


# ------------------------------------------------------------------------------------------------ #
def event_log(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        module = func.__module__
        classname = func.__qualname__

        logger = logging.getLogger(f"{module}.{classname}")

        try:
            logger.info(f"Started {module}.{classname}.")

            start_time = datetime.now()

            result = func(self, *args, **kwargs)

            end_time = datetime.now()
            duration = round((end_time - start_time).total_seconds(), 3)

            logger.info(f"Completed {module}.{classname}. Duration: {duration} seconds.")
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
            raise e

    return wrapper
