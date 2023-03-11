#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/dataset/factory.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Thursday March 9th 2023 07:20:28 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Factory Module for Datasets, Interaction Matrices and Cooccurrence Matrices"""
import pandas as pd

from recsys.data.dataset import Dataset
from recsys.operator.base import Operator, Artifact


# ------------------------------------------------------------------------------------------------ #
#                                 RATINGS DATASET FACTORY                                          #
# ------------------------------------------------------------------------------------------------ #
class DatasetFactory(Operator):
    """Dataset Factory

    Args:
        name (str): The name of the dataset
        description (str): Describes the dataset in terms of sampling strategy
        datasource (str): Defaults to 'movielens25m'
        source (str): Source file path.
        destination (str): The filepath for persisting the dataset
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        name: str,
        description: str,
        source: str,
        destination: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._name = name
        self._description = description
        self._datasource = datasource

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            dataset = Dataset(
                name=self._name,
                description=self._description,
                data=data,
                datasource=self._datasource,
            )

            self._put_data(filepath=self._destination, data=dataset)

            self._logger.info(
                f"Created {self._datasource} Dataset, {self._name} and stored at {self._destination}"
            )

            return dataset
        else:
            return self._get_data(filepath=self._destination)
