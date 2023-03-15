#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/matrix/interaction.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Saturday March 11th 2023 05:22:55 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Interaction Matrix Factory"""
from scipy.sparse import coo_matrix
import warnings

from recsys.data.dataset import Dataset
from recsys.data.matrix import Matrix
from recsys.operator.base import Operator, Artifact

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------ #
#                                 INTERACTION MATRIX FACTORY                                       #
# ------------------------------------------------------------------------------------------------ #
class InteractionMatrixFactory(Operator):
    """Interaction Matrix Factory

    The input dataset is passed through the execution method.

    Args:
        name (str): The name of the interaction matrix
        description (str): Describes the interaction matrix
        filepath (str): The filepath for persisting the matrix
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        name: str,
        description: str,
        destination: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(destination=destination, force=force)
        self._name = name
        self._description = description
        self._datasource = datasource

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def execute(self, data: Dataset, context: dict = None) -> None:
        """Creates and persists the interaction matrix object.

        Args:
            data (Dataset): Dataset object.

        """

        if not self._skip(endpoint=self._destination):

            dataset = data or self._get_data(filepath=self._source)

            df = dataset.to_df()
            df = df[["useridx", "itemidx"]]
            df["interaction"] = 1
            rows = df["useridx"]
            cols = df["itemidx"]
            data = df["interaction"]
            coo = coo_matrix((data, (rows, cols)), shape=(dataset.n_users, dataset.n_items))

            matrix = Matrix(
                name=self._name,
                description=self._description,
                data=coo,
                datasource=self._datasource,
            )

            self._put_data(filepath=self._destination, data=matrix)

            self._logger.info(
                f"Created Interaction Matrix, {self._name} and stored at {self._destination}"
            )

            return matrix
        else:
            return self._get_data(filepath=self._destination)
