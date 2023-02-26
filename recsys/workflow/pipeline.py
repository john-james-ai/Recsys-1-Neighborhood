#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/pipeline.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 09:50:07 pm                                             #
# Modified   : Sunday February 26th 2023 01:35:04 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""PIpeline Module"""
from abc import ABC, abstractmethod
import importlib
import pandas as pd

from recsys.workflow.base import Event
from recsys.workflow.task import Task
from recsys.workflow.operator import Operator


# ------------------------------------------------------------------------------------------------ #
class Pipeline(Event):
    """Tasks executed in sequence."""

    def __init__(self, name: str, description: str, *args, **kwargs) -> None:
        super().__init__(name=name, description=description)

        self._tasks = {}

    def add_task(self, task: Task) -> None:
        if task.name in self._tasks.keys():
            msg = f"Unable to add task: {task.name}. It already exists in pipeline: {self._name}."
            self._logger.error(msg)
            raise FileExistsError(msg)
        else:
            self._tasks[task.name] = task

    def remove_task(self, name: str) -> None:
        try:
            del self._tasks[name]
        except KeyError:
            msg = f"Task {name} does not exist in pipeline {self._name}."
            self._logger.error(msg)
            raise FileNotFoundError(msg)

    def print_tasks(self) -> pd.DataFrame:
        tasks = []
        for name, task in self._tasks.items():
            d = {"name": name, "description": task.description, "status": task.status}
            tasks.append(d)
        df = pd.DataFrame(data=tasks)
        print(df)

    def run(self) -> None:
        """Executes the pipeline"""
        data = None
        self._setup()

        for task in self._tasks.values():
            data = task.run(data)

        self._teardown()


# ------------------------------------------------------------------------------------------------ #
#                                       BUILDER                                                    #
# ------------------------------------------------------------------------------------------------ #
class Builder(ABC):
    """Interface definition for creating Pipelines."""

    @property
    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Returns the completed Pipelinie object"""

    @abstractmethod
    def build_pipeline(self) -> None:
        """Constructs the Pipeline object."""

    @abstractmethod
    def _build_task(self) -> Task:
        """Constructs a task object"""

    @abstractmethod
    def _build_operator(self) -> Operator:
        """Builds an operator object."""


# ------------------------------------------------------------------------------------------------ #
#                                    PIPELINE BUILDER                                              #
# ------------------------------------------------------------------------------------------------ #
class PipelineBuilder(Builder):
    """Implementation of Pipeline Builder"""

    def __init__(self, config: dict) -> None:
        self._config = config
        self.reset()

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def reset(self) -> None:
        self._pipeline = None

    def build_pipeline(self) -> None:
        """Constructs a Pipeline object"""
        config = self._config.get("pipeline")
        self._pipeline = Pipeline(name=config.get("name"), description=config.get("description"))
        task_configs = config.get("tasks")
        for task_config in task_configs:
            task = self._build_task(task_config)
            self._pipeline.add_task(task)

    def _build_task(self, task_config: dict) -> Task:
        """Builds an individual Task object."""
        config = task_config.get("task")
        operator = self._build_operator(config.get("operator"))
        task = Task(
            name=config.get("name"),
            description=config.get("description"),
            operator=operator,
        )
        return task

    def _build_operator(self, operator_config: dict) -> Operator:

        try:

            module = importlib.import_module(name=operator_config.get("module"))
            function = getattr(module, operator_config.get("name"))

            operator = function(**operator_config.get("params"))
            return operator

        except KeyError as e:
            msg = "Operator configuration is missing or malformed."
            self._logger.error(msg)
            raise (e)
