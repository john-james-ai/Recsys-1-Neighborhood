#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/builder.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 07:59:03 pm                                                #
# Modified   : Sunday March 19th 2023 02:56:54 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
from abc import ABC, abstractmethod
import importlib
import logging

from recsys.services.io import IOService
from recsys.workflow.pipeline import Pipeline

# ------------------------------------------------------------------------------------------------ #


class PipelineBuilder(ABC):
    """Constructs Configuration file based Pipeline objects"""

    def __init__(self) -> None:
        self._config = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def reset(self) -> None:
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def load_config(self, name: str, config_filepath: str) -> None:
        """Loads the Pipeline configuration from file

        Args:
            config_filepath (str): The path to the pipeline config file.
        """
        contents = IOService.read(filepath=config_filepath)
        try:
            self._config = contents[name]
        except KeyError:
            msg = (
                f"Configuration in {config_filepath} has no configuration for the {name} pipeline."
            )
            self._logger.error(msg)
            raise

    def build(self) -> None:
        """Constructs a Pipeline object."""

        pipeline = self._build_pipeline(self._config)
        steps = self._build_tasks(self._config.get("tasks", None))
        pipeline.set_steps(steps)
        self._pipeline = pipeline

    @abstractmethod
    def _build_pipeline(self, config: dict) -> Pipeline:
        """Delegated to subclasses."""

    def _build_steps(self, config: dict) -> list:
        """Iterates through task and returns a list of task objects."""

        steps = {}

        for _, step_config in config.items():

            try:

                # Create task object from string using importlib
                module = importlib.import_module(name=step_config["module"])
                step = getattr(module, step_config["operator"])

                operator = step(
                    name=step_config["name"],
                    params=step_config["params"],
                )

                steps[operator.name] = operator

            except KeyError as e:
                logging.error("Configuration File is missing operator configuration data")
                raise (e)

        return steps
