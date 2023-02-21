#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/profile.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:53:01 pm                                             #
# Modified   : Monday February 20th 2023 03:06:42 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Logging module for workflow package"""
import functools
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
import logging

import sqlalchemy.engine
from dependency_injector.wiring import Provide, inject
from recsys.container import Recsys

# ------------------------------------------------------------------------------------------------ #
#                                OPERATOR LOG DECORATOR                                            #
# ------------------------------------------------------------------------------------------------ #


@inject
def task_profile(engine: sqlalchemy.engine = Provide[Recsys.data.db]):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            module = func.__module__
            classname = func.__qualname__
            name = self.__dict__.get("_name", None)
            description = self.__dict__.get("_description", None)

            logger = logging.getLogger(f"{module}.{classname}")

            try:
                logger.info(f"Started {module}.{classname} instance name: {name} {description}")

                start_time = datetime.now()
                start_cpu_user_time = psutil.cpu_times().user
                start_cpu_system_time = psutil.cpu_times().system
                _ = psutil.cpu_percent()  # Returns 0 on first call
                start_pct_memory_used = psutil.virtual_memory().percent
                start_pct_disk_used = psutil.disk_usage("/").percent

                result = func(self, *args, **kwargs)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                end_cpu_user_time = psutil.cpu_times().user
                end_cpu_system_time = psutil.cpu_times().system
                end_cpu_percent = psutil.cpu_percent()
                end_pct_memory_used = psutil.virtual_memory().percent
                end_pct_disk_used = psutil.disk_usage("/").percent

                d = {}
                d["pid"] = psutil.users()[0].pid
                d["name"] = name
                d["operator"] = classname
                d["module"] = module
                d["description"] = description
                d["started"] = start_time
                d["completed"] = end_time
                d["duration"] = (end_time - start_time).total_seconds()
                d["cpu_user_time"] = end_cpu_user_time - start_cpu_user_time
                d["cpu_system_time"] = end_cpu_system_time - start_cpu_system_time
                d["cpu_percent"] = end_cpu_percent
                d["pct_memory_used"] = np.mean([start_pct_memory_used, end_pct_memory_used])
                d["pct_disk_used"] = np.mean(start_pct_disk_used, end_pct_disk_used)
                df = pd.DataFrame(data=d, index=[0])
                df.to_sql("profile", con=engine, if_exists="append")

                logger.info(
                    f"Completed {module}.{classname} instance name: {name}. Duration: {duration}"
                )
                return result
            except Exception as e:
                logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
                raise e

        return wrapper

    return decorator
