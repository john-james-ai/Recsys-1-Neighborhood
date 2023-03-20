#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/task.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 07:57:18 pm                                                #
# Modified   : Sunday March 19th 2023 04:13:29 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union
from datetime import datetime

import mlflow
import pandas as pd

from recsys.workflow.event import Event
from dataprep.operator import Operator
from recsys.dataset.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class Task(Event):
    """Object that performs a step in a pipeline.
    Args:
        name (str): Task name
        desc (str): desc for the task
        operator (Operator): An instance of an Operator which fulfills the task.
    """

    def __init__(
        self,
        name: str,
        desc: str,
        operator: Operator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._name = name
        self._desc = desc
        self._operator = operator

        self._state = "created"
        self._started = None
        self._ended = None
        self._duration = None

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> Union[None, pd.DataFrame, Dataset]:
        """Runs the task."""

        self._setup()
        data = self._operator.__call__(data)
        self._teardown()
        return data

    @property
    def name(self) -> str:
        """Returns the name of the task"""
        return self._name

    @property
    def desc(self) -> str:
        """Returns the desc of the task"""
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

    def _teardown(self, artifact: dict = None) -> None:
        """Wrap up activities."""
        self._ended = datetime.now()
        self._state = "success"
        self._duration = (self._ended - self._started).total_seconds()
        mlflow.log_metric("duration", self._duration)
        self._log_artifact()
        self._logger.info(
            f"Completed task: {self._name}. Duration: {round(self._duration,2)} seconds."
        )

    def _log_artifact(self) -> None:
        """Logs the artifact in MLFlow."""
        if self._operator.artifact is not None:
            if self._operator.artifact.isfile:
                mlflow.log_artifact(
                    local_path=self._operator.artifact.path,
                    artifact_path=self._operator.artifact.uripath,
                )
            else:
                mlflow.log_artifacts(
                    local_dir=self._operator.artifact.path,
                    artifact_path=self._operator.artifact.uripath,
                )
