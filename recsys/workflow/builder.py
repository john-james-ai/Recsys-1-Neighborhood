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
# Modified   : Monday March 20th 2023 01:23:52 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
from abc import ABC, abstractmethod
import importlib
import logging

from dependency_injector.wiring import Provide, inject

from recsys.container import Recsys
from recsys.services.io import IOService
from recsys.asset.base import Asset
from recsys.asset.centre import AssetCentre
from recsys.workflow.pipeline import Pipeline
from recsys.workflow.task import Task

# ------------------------------------------------------------------------------------------------ #

@inject
class PipelineBuilder(ABC):
    """Constructs Configuration file based Pipeline objects"""

    def __init__(self, asset_centre: AssetCentre=Provide[Recsys.asset.centre]) -> None:
        self._config = None
        self._asset_centre = asset_centre
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
            name (str): The name of the pipeline which will be matched against the dictionary
                key in the config file.
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

        self._build_pipeline(self._config)
        for task_config in self._config.get('tasks'):
            task = self._build_task(task_config)


    @abstractmethod
    def _build_pipeline(self, config: dict) -> Pipeline:
        """Instantiates a Pipeline object."""
        self._pipeline = Pipeline(name=config.get('name'), desc=config.get('desc'))

    def _build_task(self, config: dict) -> Task:
        """Constructs a task and returns it."""
        task_input = self._get_task_input(config)
        task = Task(name=config.get('name'), desc=config.get('desc'))

    def _get_task_input(self, config: dict) -> Asset:
        """Reads the input schema and obtains the asset from persistence."""
        input_schema = config.get('input_schema')
        asset_type = input_schema.get('type')
        asset_name = input_schema.get('name')
        asset = self._asset_centre.get(name=asset_name, asset_type=asset_type)

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
