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
# Modified   : Friday March 17th 2023 03:00:23 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
import importlib
from datetime import datetime
import mlflow

from recsys import Operator
from recsys.workflow.base import Event, Builder
from recsys.workflow.task import Task
from recsys.services.io import IOService
from recsys.services.data_types import RecursiveNamespace

# ------------------------------------------------------------------------------------------------ #


class Pipeline(Event):
    """Pipeline class, a collection of tasks executed in a sequence.
    Args:
        name (str): Human readable name for the pipeline run.
        description (str): Description of the pipelne
        context (dict): Data required by all operators in the pipeline. Optional.
    """

    def __init__(self, name: str, description: str, context: dict = None) -> None:
        super().__init__()
        self._name = name
        self._description = description
        self._context = context
        self._tasks = {}
        self._started = None
        self._ended = None
        self._duration = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
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

    def _setup(self) -> None:
        """Performs required initialization steps before running the task"""
        self._started = datetime.now()
        mlflow.start_run(
            run_name=self._name,
            nested=False,
        )
        self._logger.info(f"Started task: {self._name} ")

    def _teardown(self) -> None:
        """Wrap up activities."""
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        mlflow.log_metric("duration", self._duration)
        mlflow.end_run()
        self._logger.info(
            f"Completed task: {self._name}. Duration: {round(self._duration,2)} seconds."
        )

    def add_task(self, task: Event) -> None:
        self._tasks[task.name] = task

    def run(self) -> None:
        """Runs the pipeline"""
        data = None
        self._setup()

        for task in self._tasks.values():
            data = task.run(data, self._context)

        self._teardown()

    def check(self) -> bool:
        """Checks the readiness of the pipeline and returns True if ready, False otherwise."""
        self._logger.debug(f"Pipeline has {len(self._tasks)} tasks.")
        for i, (name, task) in enumerate(self._tasks.items(), start=1):
            if isinstance(task, Event):
                self._logger.debug(
                    f"\n\nTask: {i}\tClass: {task.__class__.__name__}\n\tOperator: {task.operator.__class__.__name__}\n\tName: {task.name}\n\tDescription: {task.description}\n"
                )
            else:  # pragma: no cover
                self._logger.error("Pipeline error: Tasks are not subclasses of Event.")
                raise Exception
        return True


# ------------------------------------------------------------------------------------------------ #


class PipelineBuilder(Builder):
    """Constructs Configuration file based Pipeline objects"""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:  # pragma: no cover
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def build(self, config_filepath: str) -> None:
        """Constructs a Pipeline object.
        Args:
            config_filepath (str): Pipeline configuration
        """

        config = self._build_config(config_filepath)
        self._build_pipeline(config)
        self._build_tasks(config)
        self._logger.info(f"Pipeline {config.name} construction is complete.")

    def _build_config(self, config_filepath: str) -> RecursiveNamespace:
        """Reads the config file and returns a recursive namespace object."""
        config = IOService.read(config_filepath)
        return RecursiveNamespace(**config).pipeline

    def _build_pipeline(self, config) -> None:
        """Initialize a Pipeline object."""
        self._pipeline = Pipeline(name=config.name, description=config.description)

    def _build_tasks(self, config) -> list:
        """Iterates through task and returns a list of task objects."""

        # Iteratively build the tasks and add them to the pipeline.
        for task_config in config.tasks:
            task = self._build_task(task_config)
            self._pipeline.add_task(task)

    def _build_task(self, task_config) -> Task:
        """Builds a task object."""
        operator = self._build_operator(task_config.operator)
        return Task(name=task_config.name, description=task_config.description, operator=operator)

    def _build_operator(self, operator_config) -> Operator:
        """Constructs the operator object that executes the task."""

        try:

            # Create task object from string using importlib
            module = importlib.import_module(name=operator_config.module)
            callable = getattr(module, operator_config.name)

            operator = callable(**vars(operator_config.params))

            return operator

        except KeyError as e:  # pragma: no cover
            self._logging.error("Configuration File is missing operator configuration data")
            raise (e)
