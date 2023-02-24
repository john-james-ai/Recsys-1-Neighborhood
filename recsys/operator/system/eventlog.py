#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/system/eventlog.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 09:53:01 pm                                             #
# Modified   : Thursday February 23rd 2023 03:44:25 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Logging module for workflow package"""
import functools
import psutil
import pandas as pd
import cProfile
import numpy as np
from datetime import datetime
import logging

from dependency_injector.wiring import Provide, inject
from recsys.container import Recsys

# ------------------------------------------------------------------------------------------------ #
#                                OPERATOR PROFILER                                                 #
# ------------------------------------------------------------------------------------------------ #


@inject
def eventlog(con=Provide[Recsys.db.db]):
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
                start_iowait = psutil.cpu_times().iowait
                start_steal = psutil.cpu_times().steal
                _ = psutil.cpu_percent()  # Returns 0 on first call
                start_cpu_freq_min = psutil.cpu_freq().min
                start_cpu_freq_max = psutil.cpu_freq().max
                start_total_memory = psutil.virtual_memory().total
                start_avail_memory = psutil.virtual_memory().available
                start_active_mem = psutil.virtual_memory().active
                start_inactive_mem = psutil.virtual_memory().inactive
                start_shared_mem = psutil.virtual_memory().shared
                start_pct_memory_used = psutil.virtual_memory().percent
                start_total_swap_memory = psutil.swap_memory().total
                start_used_swap_memory = psutil.swap_memory().used
                start_free_swap_memory = psutil.swap_memory().free
                start_pct_swap_memory_used = psutil.swap_memory().percent
                start_pct_disk_used = psutil.disk_usage("/").percent
                start_read_count = psutil.disk_io_counters().read_count
                start_write_count = psutil.disk_io_counters().write_count

                result = func(self, *args, **kwargs)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                end_cpu_user_time = psutil.cpu_times().user
                end_cpu_system_time = psutil.cpu_times().system
                end_iowait = psutil.cpu_times().iowait
                end_steal = psutil.cpu_times().steal
                end_cpu_percent = psutil.cpu_percent()
                end_cpu_freq_min = psutil.cpu_freq().min
                end_cpu_freq_max = psutil.cpu_freq().max
                end_total_memory = psutil.virtual_memory().total
                end_avail_memory = psutil.virtual_memory().available
                end_active_mem = psutil.virtual_memory().active
                end_inactive_mem = psutil.virtual_memory().inactive
                end_shared_mem = psutil.virtual_memory().shared
                end_pct_memory_used = psutil.virtual_memory().percent
                end_total_swap_memory = psutil.swap_memory().total
                end_used_swap_memory = psutil.swap_memory().used
                end_free_swap_memory = psutil.swap_memory().free
                end_pct_swap_memory_used = psutil.swap_memory().percent
                end_pct_disk_used = psutil.disk_usage("/").percent
                end_read_count = psutil.disk_io_counters().read_count
                end_write_count = psutil.disk_io_counters().write_count

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
                d["cpu_freq_min"] = min(start_cpu_freq_min, end_cpu_freq_min)
                d["cpu_freq_max"] = min(start_cpu_freq_max, end_cpu_freq_max)
                d["iowait"] = end_iowait - start_iowait
                d["oswait"] = end_steal - start_steal
                d["memory_total_ave"] = np.mean(end_total_memory, start_total_memory)
                d["memory_avail_ave"] = np.mean(end_avail_memory, start_avail_memory)
                d["pct_memory_used"] = np.mean([start_pct_memory_used, end_pct_memory_used])
                d["ave_memory_avail"] = np.mean([start_pct_memory_used, end_pct_memory_used])
                d["ave_active_memory"] = np.mean(start_active_mem, end_active_mem)
                d["ave_inactive_memory"] = np.mean(start_inactive_mem, end_inactive_mem)
                d["ave_shared_memory"] = np.mean(start_shared_mem, end_shared_mem)
                d["ave_total_swap_memory"] = np.mean(start_total_swap_memory, end_total_swap_memory)
                d["ave_used_swap_memory"] = np.mean(start_used_swap_memory, end_used_swap_memory)
                d["ave_free_swap_memory"] = np.mean(start_free_swap_memory, end_free_swap_memory)
                d["ave_pct_swap_memory_used"] = np.mean(
                    start_pct_swap_memory_used, end_pct_swap_memory_used
                )
                d["pct_disk_used"] = np.mean(start_pct_disk_used, end_pct_disk_used)
                d["read_count"] = end_read_count - start_read_count
                d["write_count"] = end_write_count - start_write_count

                df = pd.DataFrame(data=d, index=[0])
                df.to_sql("profile", con=con, if_exists="append")

                logger.info(
                    f"Completed {module}.{classname} instance name: {name}. Duration: {duration}"
                )
                return result
            except Exception as e:
                logger.exception(f"Exception raised in {module}.{classname}. Exception: {str(e)}")
                raise e

        return wrapper

    return decorator


# ------------------------------------------------------------------------------------------------ #
#                                OPERATOR PROFILER 2                                               #
# ------------------------------------------------------------------------------------------------ #


def profiler(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            # Note use of name from outer scope
            prof.dump_stats(name)
            return retval

        return wrapper

    return inner
