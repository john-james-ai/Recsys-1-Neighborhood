#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/assets/experiment.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 02:36:54 am                                                #
# Modified   : Wednesday March 1st 2023 07:07:55 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime

from recsys import Asset
from recsys.data.dataset import Dataset
from recsys.assets.model import Model


# ------------------------------------------------------------------------------------------------ #
#                                  EXPERIMENT CLASS                                                #
# ------------------------------------------------------------------------------------------------ #
class ExperimentAsset(Asset):  # pragma: no cover
    """Base class for experiments.
    Args:
        name (str): Name of the experiment
        description (str): Description for the experiment
        dataset (Any): The dataset used in the experiment
        model (Model): The model used in the experiment

    """

    def __init__(self, name: str, description: str, dataset: Dataset, model: Model) -> None:
        super().__init__(name=name, description=description)
        self._dataset = dataset
        self._model = model
        self._result = None

    @abstractmethod
    def run(self) -> Result:
        """Runs the experiment."""
        pass


@dataclass
class Result:
    name: str  # Experiment name
    description: str  # Experiment description
    dataset: Dataset  # Dataset used in the experiment
    model: Model  # The model used in the experiment
    timestamp: datetime  # The datetime the experiment was run.
    training_duration: int  # The time take to train the model in seconds
    prediction_time: int  # The time to perform the predictions in seconds.
    metric: str  # The prediction metric
    score: int  # The score from the prediction
