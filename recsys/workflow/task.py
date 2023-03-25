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
# Modified   : Monday March 20th 2023 08:31:19 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union
from datetime import datetime
import logging
from typing import Union, Any

import mlflow
import pandas as pd
from dependency_injector.wiring import Provide, inject

from recsys.asset.base import Asset
from recsys.workflow.event import Event
from recsys import Operator
from recsys.dataset.base import Dataset
from recsys.container import Recsys
from recsys.asset.centre import AssetCentre
from recsys.asset.factory import AssetFactory
from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
@inject
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
        input_schema: dict,
        output_schema: dict,
        operator: Operator,
        assets: AssetCentre = Provide[Recsys.assets.centre],
        io: IOService = IOService,
        *args,
        **kwargs,
    ) -> None:
        self._name = name
        self._desc = desc
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._operator = operator
        self._assets = assets
        self._io = io

        self._state = "created"
        self._started = None
        self._ended = None
        self._duration = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> Union[None, pd.DataFrame, Dataset]:
        """Runs the task."""

        self._setup()
        data = self._operator.__call__(data)
        self._setdown()
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

    def _setup(self) -> Union[Any, None]:
        """Performs required initialization steps and obtains input data before running the task"""
        self._logger.info(f"Started task: {self._name} ")
        self._started = datetime.now()
        self._state = "running"
        return self._get_input(self._input_schema)

    def _setdown(self, data: Asset = None) -> None:
        """Wrap up activities, including persisting of assets, and logging artifacts."""
        self._persist_asset(asset)
        self._log_asset(asset)
        self._ended = datetime.now()
        self._state = "success"
        self._duration = (self._ended - self._started).total_seconds()
        mlflow.log_metric("duration", self._duration)
        self._log_artifact()
        self._logger.info(
            f"Completed task: {self._name}. Duration: {round(self._duration,2)} seconds."
        )

    def _get_input(self) -> Any:
        """Obtains Task input based upon input schema."""

    def _create_asset(self, **kwargs: dict) -> Asset:
        """Creates an Asset from the data returned from the operator, if any."""
        return AssetFactory.build(self._output_schema, kwargs)

    def _put_output(self, data: Any) -> None:
        """Persists output according to output schema."""

    def _log_output(self) -> None:
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
