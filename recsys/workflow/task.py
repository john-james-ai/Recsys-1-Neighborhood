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
# Created    : Saturday March 4th 2023 08:14:42 pm                                                 #
# Modified   : Friday March 17th 2023 03:00:23 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Any, Union
from datetime import datetime

import mlflow

from recsys.workflow.base import Event
from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
class Task(Event):
    """Object that performs a step in a pipeline.

    Args:
        name (str): Task name
        description (str): Description for the task
        operator (Operator): An instance of an Operator which fulfills the task.
    """

    def __init__(
        self,
        name: str,
        description: str,
        operator: Operator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._operator = operator

        self._started = None
        self._ended = None
        self._duration = None

    @property
    def name(self) -> str:
        """Returns the name of the task"""
        return self._name

    @property
    def description(self) -> str:
        """Returns the description of the task"""
        return self._description

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
    def operator(self) -> Operator:
        return self._operator

    def _setup(self) -> None:
        """Performs required initialization steps before running the task"""
        self._started = datetime.now()
        mlflow.start_run(
            run_name=self._name,
            nested=True,
        )
        self._logger.info(f"Started task: {self._name} ")

    def _teardown(self, artifact: dict = None) -> None:
        """Wrap up activities."""
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        mlflow.log_metric("duration", self._duration)
        self._log_artifact()
        mlflow.end_run()
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

    def run(self, data: Any = None, context: dict = None) -> Union[None, Any]:
        """Runs the task."""

        self._setup()
        data = self._operator.execute(data, context)
        self._teardown()
        return data
