#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/pipeline_builder.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 07:59:03 pm                                                #
# Modified   : Saturday March 18th 2023 08:02:00 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
import os
from abc import ABC, abstractmethod
import importlib
import logging

from recsys.services.io import IOFactory
from recsys.workflow.pipeline import Pipeline

# ------------------------------------------------------------------------------------------------ #


class PipelineBuilder(ABC):
    """Constructs Configuration file based Pipeline objects"""

    def reset(self) -> None:
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def build(self, config_filepath: str) -> None:
        """Constructs a Pipeline object.
        Args:
            config_filepath (str): Pipeline configuration
        """
        config = self._get_config(config_filepath)
        pipeline = self.build_pipeline(config)
        steps = self._build_steps(config.get("steps", None))
        pipeline.set_steps(steps)
        self._pipeline = pipeline

    def _get_config(self, config_filepath: str) -> dict:
        fileformat = os.path.splitext(config_filepath)[1].replace(".", "")
        io = IOFactory.io(fileformat=fileformat)
        return io.read(config_filepath)

    @abstractmethod
    def build_pipeline(self, config: dict) -> Pipeline:
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
