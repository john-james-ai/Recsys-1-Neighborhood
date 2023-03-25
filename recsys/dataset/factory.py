#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataset/factory.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 20th 2023 04:18:19 am                                                  #
# Modified   : Monday March 20th 2023 06:28:20 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Dataset Factory Module"""
import logging

import pandas as pd

from recsys.dataset.movielens import MovieLens
from recsys.workflow.operator import Operator
from recsys.dataset.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class DatasetFactoryOperator(Operator):
    """Creates a Dataset subclass object.

    Args:
        name (str): The name of the Dataset object to create
        desc (str): The dataset description
        dataset_type (str): The class name for the Dataset subclass to create
    """

    __name = "dataset_factory_operator"
    __desc = "Creates Dataset objects of the specified dataset (sub) type."

    __datasets = {"movielens": MovieLens}

    def __init__(self, name: str, desc: str, dataset_type: str) -> None:
        self._name = name
        self._desc = desc
        self._dataset_type = dataset_type
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __call__(self, data: pd.DataFrame) -> Dataset:
        """Creates a Dataset object of the appropriate subclass."""
        try:
            dataset = DatasetFactoryOperator.__datasets[self._dataset_type.lower()]
            return dataset(name=self._name, desc=self._desc, data=data)
        except KeyError:
            msg = f"{self._dataset_type} is not a valid Dataset type."
            self._logger.error(msg)
            raise TypeError(msg)
