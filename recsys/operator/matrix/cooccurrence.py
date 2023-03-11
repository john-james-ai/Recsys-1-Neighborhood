#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/matrix/cooccurrence.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Thursday March 9th 2023 07:34:44 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cooccurrence Matrix Factory"""

from recsys.data.matrix import Matrix
from recsys.operator.base import Operator, Artifact

# ------------------------------------------------------------------------------------------------ #


class CooccurrenceMatrixFactory(Operator):
    """Cooccurrence Matrix Factory

    Args:
        name (str): The name of the interaction matrix
        description (str): Describes the interaction matrix
        destination (str): The filepath for persisting the matrix
        axis (str): 0 for row or user, 1 for column or item
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        name: str,
        description: str,
        destination: str,
        axis: int,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(destination=destination, force=force)
        self._name = name
        self._description = description
        self._datasource = datasource
        self._axis = axis

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def execute(self, data: Matrix, context: dict = None) -> None:
        """Creates and persists the cooccurrence matrix object from an interaction matrix.

        Args:
            data (Matrix): The Interaction Matrix

        """

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)
            if self._axis == 0:
                csr = data.to_csr()
                csrT = csr.transpose()
                cooccurrence = csr.dot(csrT).tocoo()
            else:
                csc = data.to_csc()
                cscT = csc.transpose()
                cooccurrence = cscT.dot(csc).tocoo()

            matrix = Matrix(
                name=self._name,
                description=self._description,
                data=cooccurrence,
                datasource=self._datasource,
            )

            self._put_data(filepath=self._destination, data=matrix)

            self._logger.info(
                f"Created Cooccurrence Matrix, {self._name} and stored at {self._destination}"
            )

            return matrix
        else:
            return self._get_data(filepath=self._destination)
