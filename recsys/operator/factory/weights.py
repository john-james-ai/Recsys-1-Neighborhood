#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/factory/weights.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Saturday March 18th 2023 08:41:29 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cooccurrence Matrix Factory"""
from recsys.matrix.base import Matrix
from recsys import Operator, Artifact

# ------------------------------------------------------------------------------------------------ #


class SignificanceWeightedMatrixFactory(Operator):
    """Significance Weighting

    Here we compute a significance weighting proportional to the number of common items or users
    which have rated an item.

    The weight is computed as follows:
    Wuv = min(|Iuv|, threshold) / threshold

    Weights computed from an interaction matrix will be multiplied element-wise by the similarity matrix
    as follows:
    - User: Wuv * Suv = S'uv
    - Item: Wij * Sij = S'ij

    Where Suv and Sij are user similarity and item similarity matrices, respectively.

    The similarity matrix will be read from file as the source. An interaction matrix used to compute
    the weights will be passed into the __call__ method

    Args:
        name (str): The name of the weighted similarity matrix
        description (str): Describes the weighted similarity matrix
        source (str): The filepath of the similarity matrix.
        destination (str): The filepath to the weighted similarity matrix.
        dim (str): Either 'u' or 'user' for user dimension, or 'i' or 'item' for item dimension.
        threshold (float): Value > 0. Default is 50.
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
        threshold: int = 50,
        datasource="movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._name = name
        self._threshold = threshold
        self._description = description
        self._datasource = datasource

        try:
            self._dim = SignificanceWeightedMatrixFactory.__dims[dim[0].lower()]
        except KeyError:
            msg = f"dim parameter value {dim} is not supported. Valid values are: {SignificanceWeightedMatrixFactory.__dims}"
            self._logger.error(msg)
            raise ValueError(msg)

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def __call__(self, data: Matrix, context: dict = None) -> Matrix:
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
        Wuv = Iuv.minimum(self._threshold) / self._threshold

        # Extract user similarity from the Matrix object
        Suv = similarity.to_csr()

        # Apply weight to similarity
        Suv = Wuv.multiply(Suv)

        matrix = Matrix(
            name=self._name,
            description=self._description,
            data=Suv,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=matrix)

        return matrix

    def _compute_item_weights(self, interactions: Matrix, similarity: Matrix) -> Matrix:
        """Computes item weights"""

        # Compute item weights.
        csc = interactions.to_csc()
        Uij = csc.T.dot(csc)
        Wij = Uij.minimum(self._threshold) / self._threshold

        # Extract item similarity from the Matrix object
        Sij = similarity.to_csc()

        # Apply weight
        Sij = Wij.multiply(Sij)

        matrix = Matrix(
            name=self._name,
            description=self._description,
            data=Sij,
            datasource=self._datasource,
        )

        self._put_data(filepath=self._destination, data=matrix)

        return matrix
