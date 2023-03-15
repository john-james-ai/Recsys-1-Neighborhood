#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/factory/weights.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Sunday March 12th 2023 03:54:21 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cooccurrence Matrix Factory"""
import numpy as np
from scipy.sparse import coo_matrix

from recsys.data.matrix import Matrix
from recsys.data.array import Array
from recsys.operator.base import Operator, Artifact

# ------------------------------------------------------------------------------------------------ #


class SignificanceWeightedMatrixFactory(Operator):
    """Significance Weighting

    Here we compute a significance weighting proportional to the number of common items or users
    which have rated an item.

    The weight is computed as follows:
    Wuv = min(|Iuv|, gamma) / gamma

    Weights computed from an interaction matrix will be multiplied element-wise by the similarity matrix
    as follows:
    - User: Wuv * Suv = S'uv
    - Item: Wij * Sij = S'ij

    Where Suv and Sij are user similarity and item similarity matrices, respectively.

    The similarity matrix will be read from file as the source. An interaction matrix used to compute
    the weights will be passed into the execute method

    Args:
        name (str): The name of the weighted similarity matrix
        description (str): Describes the weighted similarity matrix
        source (str): The filepath of the similarity matrix.
        destination (str): The filepath to the weighted similarity matrix.
        dim (str): Either 'u' or 'user' for user dimension, or 'i' or 'item' for item dimension.
        gamma (float): Value > 0. Default is 25.
        datasource (str): The source of the dataset. Default = 'movielens25m'.
        force (bool): Whether to overwrite existing data if it already exists.

    """

    __dims = {"u": "User", "i": "Item"}

    def __init__(
        self,
        name: str,
        description: str,
        source: str,
        destination: str,
        dim: str,
        gamma: int = 30,
        datasource="movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._name = name
        self._gamma = gamma
        self._description = description
        self._datasource = datasource

        try:
            self._dim = SignificanceWeightedMatrixFactory.__dims[dim[0].lower()]
        except KeyError:
            msg = f"dim parameter value {dim} is not supported. Valid values are: {SignificanceWeightedMatrixFactory.__dims}"
            self._logger.error(msg)
            raise ValueError(msg)

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def execute(self, data: Matrix, context: dict = None) -> Matrix:
        """Creates a signficance weighting matrix from an interaction matrix.

        Args:
            data (Matrix): The Interaction Matrix

        """

        if not self._skip(endpoint=self._destination):

            similarity = data

            interactions = self._get_data(filepath=self._source)
            if self._dim[0].lower() == "u":
                return self._compute_user_weights(interactions, similarity)
            else:
                return self._compute_item_weights(interactions, similarity)

        else:
            return self._get_data(filepath=self._destination)

    def _compute_user_weights(self, interactions: Matrix, similarity: Matrix) -> Matrix:
        """Computes user weights"""

        # Compute user weights
        csr = interactions.to_csr()
        Iuv = csr.dot(csr.T)
        Wuv = Iuv.minimum(self._gamma) / self._gamma

        # Extract user similarity from the Matrix object
        Suv = similarity.to_csr()

        # Apply weight to similarity
        Suv = Wuv.multiply(Suv)

        # Convert to COO format
        coo = coo_matrix(Suv)

        matrix = Matrix(
            name=self._name,
            description=self._description,
            data=coo,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=matrix)

        return matrix

    def _compute_item_weights(self, interactions: Matrix, similarity: Matrix) -> Matrix:
        """Computes item weights"""

        # Compute item weights.
        csc = interactions.to_csc()
        Uij = csc.T.dot(csc)
        Wij = Uij.minimum(self._gamma) / self._gamma

        # Extract item similarity from the Matrix object
        Sij = similarity.to_csc()

        # Apply weight
        Sij = Wij.multiply(Sij)

        # Convert to COO format
        coo = coo_matrix(Sij)

        matrix = Matrix(
            name=self._name,
            description=self._description,
            data=coo,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=matrix)

        return matrix


# ------------------------------------------------------------------------------------------------ #


class FrequencyWeightedMatrixFactory(Operator):
    """Frequency Weighted Matrix

    Here we compute a frequency weighting proportional the log of the inverse user or item frequency.

    Args:
        name (str): The name of the weighted similarity matrix
        description (str): Describes the weighted similarity matrix
        source (str): The filepath for the similarity matrix.
        destination (str): The filepath to the weighted similarity matrix.
        dim (str): Either 'u' or 'user' for user dimension, or 'i' or 'item' for item dimension.
        datasource (str): The source of the dataset. Default = 'movielens25m'.
        force (bool): Whether to overwrite existing data if it already exists.

    """

    __dims = {"u": "User", "i": "Item"}

    def __init__(
        self,
        name: str,
        description: str,
        source: str,
        destination: str,
        dim: str,
        datasource="movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._name = name
        self._description = description
        self._datasource = datasource

        try:
            self._dim = FrequencyWeightedMatrixFactory.__dims[dim[0].lower()]
        except KeyError:
            msg = f"dim parameter value {dim} is not supported. Valid values are: {FrequencyWeightedMatrixFactory.__dims}"
            self._logger.error(msg)
            raise ValueError(msg)

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def execute(self, data: Matrix, context: dict = None) -> Matrix:
        """Creates a signficance weighting matrix from an interaction matrix.

        Args:
            data (Matrix): The Interaction Matrix

        """

        if not self._skip(endpoint=self._destination):

            interactions = data

            similarity = self._get_data(filepath=self._source)

            if self._dim[0].lower() == "u":
                return self._compute_user_weights(interactions, similarity)
            else:
                return self._compute_item_weights(interactions, similarity)

        else:
            return self._get_data(filepath=self._destination)

    def _compute_user_weights(self, interactions: Matrix, similarity: Matrix) -> Matrix:
        """Computes user weights"""

        # Compute the ratio of users to the number of users who've rated i for each i.
        csr = interactions.to_csr()
        U = csr.shape[0]  # num users
        Ui = csr.sum(axis=0)  # num users rating each item.
        Wi = np.log(U / Ui)[0, :].T

        # Extract user similarity
        Suv = similarity.to_csr()

        # Apply the weights
        Suv = Wi.sum(axis=0)

        array = Array(
            name=self._name,
            description=self._description,
            data=Wi,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=array)

        return array

    def _compute_item_weights(self, interactions: Matrix, similarity: Matrix) -> Matrix:
        """Computes item weights"""

        csc = interactions.to_csc()
        I = csc.shape[1]  # noqa: E741 num Items
        Iu = csc.sum(axis=1)  # Number of items rated by each user
        Wu = np.log(I / Iu)[:, 0]

        array = Array(
            name=self._name,
            description=self._description,
            data=Wu,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=array)

        return array
