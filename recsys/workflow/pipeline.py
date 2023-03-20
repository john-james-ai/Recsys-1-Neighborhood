#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/pipeline.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 07:31:37 pm                                                 #
# Modified   : Sunday March 19th 2023 04:13:29 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
from abc import abstractmethod
from datetime import datetime
import logging

from recsys import Dataset
from recsys.workflow.task import Task
from recsys.workflow.event import Event


# ------------------------------------------------------------------------------------------------ #


class Pipeline(Event):
    """Abstract base class for Pipelines
    Args:
        name (str): Human readable name for the pipeline run.
        desc (str): desc of the pipelne
    """

    def __init__(self, name: str, desc: str) -> None:
        self._name = name
        self._desc = desc

        self._input_schema = None
        self._output_schema = None
        self._tasks = {}
        self._state = "created"
        self._started = None
        self._ended = None
        self._duration = None

        self._logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        return self._desc

    @property
    def started(self) -> datetime:
        """Returns the datetime the task started."""
        return self._started

    @property
    def ended(self) -> datetime:
        """Returns the datetime the task ended."""
        return self._ended

    @property
    def duration(self) -> datetime:
        """Returns the duration of the task."""
        return self._duration

    @property
    def state(self) -> str:
        return self._state

    def _setup(self) -> None:
        """Performs required initialization steps before running the task"""
        self._started = datetime.now()
        self._state = "running"
        self._logger.info(f"Started task: {self._name} ")

    def _teardown(self) -> None:
        """Wrap up activities."""
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        self._state = "success"
        self._logger.info(
            f"Completed task: {self._name}. Duration: {round(self._duration,2)} seconds."
        )

    def add_task(self, task: Task) -> None:
        self._tasks[task.name] = task

    @abstractmethod
    def run(self, data: Dataset) -> None:
        """Runs the pipeline"""
