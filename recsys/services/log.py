#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/services/log.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 07:33:51 am                                                #
# Modified   : Saturday March 18th 2023 07:39:46 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import functools
import logging

from recsys.services.datetime import Timer


# ------------------------------------------------------------------------------------------------ #


def log_start(logger: str, classname: str, timer: Timer):

    date = timer.started.strftime("%m/%d/%Y")
    time = timer.started.strftime("%H:%M:%S")

    msg = "Started {} at {} on {}".format(classname, time, date)
    logger.info(msg)


def log_end(logger: str, classname: str, timer: Timer):

    date = timer.stopped.strftime("%m/%d/%Y")
    time = timer.stopped.strftime("%H:%M:%S")
    duration = timer.duration.as_string()

    msg = "Completed {} at {} on {}. Duration: {}.".format(classname, time, date, duration)
    logger.info(msg)


def log(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        module = func.__module__
        classname = func.__qualname__
        logger = logging.getLogger(f"{module}.{classname}")

        try:
            timer = Timer()
            timer.start()
            log_start(logger, classname, timer)
            result = func(self, *args, **kwargs)
            timer.stop()
            log_end(logger, classname, timer)
            return result

        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e

    return wrapper
