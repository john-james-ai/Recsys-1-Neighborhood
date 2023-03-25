#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/operator/factory/similarity.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 04:26:15 pm                                                 #
# Modified   : Sunday March 19th 2023 10:48:14 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cooccurrence Matrix Factory"""
import os
from abc import abstractmethod
from typing import Union

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from recsys.dataset.base import Dataset
from recsys.matrix.i2 import Matrix
from recsys import Operator, Artifact

# ------------------------------------------------------------------------------------------------ #


class SimilarityMatrixFactory(Operator):
    """Base Similarity Matrix Factory Strategy

    Args:
        name (str): The name of the interaction matrix
        desc (str): Describes the interaction matrix
        destination (str): The directory for persisting the matrix
        dim (str): Either 'u' or 'user' for user dimension, or 'i' or 'item' for item dimension.
        force (bool): Whether to overwrite existing data if it already exists.
        datasource (str): The original source of the data.

    """

    __dims = {"u": "User", "i": "Item"}

    def __init__(
        self,
        name: str,
        desc: str,
        destination: str,
        dim: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(destination=destination, force=force)
        self._name = name
        self._desc = desc
        self._datasource = datasource
        self._filepath = None

        try:
            self._dim = SimilarityMatrixFactory.__dims[dim[0].lower()]
        except KeyError:
            msg = f"dim parameter value {dim} is not supported. Valid values are: {SimilarityMatrixFactory.__dims}"
            self._logger.error(msg)
            raise ValueError(msg)

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="matrix")

    def __call__(self, data: Dataset, context: dict = None) -> Matrix:
        """Creates and persists a similarity matrix object from a Dataset.

        Args:
            data (Dataset): The Dataset Object

        """
        self._set_filepath()

        if not self._skip(endpoint=self._filepath):

            # Returns a csr (for user cosign similarity) or csc (for item similarity matrices.)
            sparse = self._get_sparse_matrix(data)
            # Computes and returns the cosign similarity Matrix object.
            matrix = self._compute_similarity(sparse)
            # Persist the data
            self._put_data(filepath=self._filepath, data=matrix)

            return matrix
        else:
            # Returns it if it already exists.
            return self._get_data(filepath=self._filepath)

    @abstractmethod
    def _get_sparse_matrix(self, data: Dataset) -> Union[csc_matrix, csr_matrix]:
        """Returns the sparse matrix representation of the data."""

    @abstractmethod
    def _set_filepath(self) -> None:
        """Sets the filepath for the Matrix object in the destination directory."""

    def _compute_similarity(self, matrix: Union[csc_matrix, csr_matrix]) -> Matrix:
        """Computes similarity and returns a Matrix object"""

        squared_norm = matrix.multiply(matrix)

        sum_squared_norm = np.array(squared_norm.sum(axis=1))[:, 0]

        norm = np.array(np.sqrt(sum_squared_norm))

        row_indices, _ = matrix.nonzero()

        matrix.data /= norm[row_indices]

        sim = matrix.dot(matrix.T)

        matrix = Matrix(
            name=self._name,
            desc=self._desc,
            data=sim,
            datasource=self._datasource,
        )

        return matrix


# ------------------------------------------------------------------------------------------------ #
#                                   COSINE SIMILARITY                                              #
# ------------------------------------------------------------------------------------------------ #


class CosineSimilarityMatrixFactory(SimilarityMatrixFactory):

    __filenames = {"u": "cosine_similarity_user.pkl", "i": "cosine_similarity_item.pkl"}

    def __init__(
        self,
        name: str,
        desc: str,
        destination: str,
        dim: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            desc=desc,
            destination=destination,
            dim=dim,
            datasource=datasource,
            force=force,
        )
        self._filepath = None

    def _get_sparse_matrix(self, data: Matrix) -> Union[csc_matrix, csr_matrix]:
        """Obtains the sparse matrix for user or item dimensions.

        Args::
            data (Dataset): user item ratings Dataset object.

        """

        if "u" in self._dim.lower():
            return data.to_csr()
        else:
            return data.to_csc().T

    def _set_filepath(self) -> None:
        """Sets the filepath for the Matrix object in the destination directory."""
        filename = CosineSimilarityMatrixFactory.__filenames[self._dim[0].lower()]
        self._filepath = os.path.join(self._destination, filename)


# ------------------------------------------------------------------------------------------------ #
#                                 ADJUSTED COSINE SIMILARITY                                       #
# ------------------------------------------------------------------------------------------------ #
class AdjustedCosineSimilarityMatrixFactory(SimilarityMatrixFactory):

    __filenames = {
        "u": "adjusted_cosine_similarity_user.pkl",
        "i": "adjusted_cosine_similarity_item.pkl",
    }

    def __init__(
        self,
        name: str,
        desc: str,
        destination: str,
        dim: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            desc=desc,
            destination=destination,
            dim=dim,
            datasource=datasource,
            force=force,
        )

    def _get_sparse_matrix(self, data: Matrix) -> Union[csc_matrix, csr_matrix]:
        """Obtains the sparse matrix for user or item dimensions.

        For user similarity, ratings are centered by item average ratings and
        vice-versa for item similarity.

        Args::
            data (Dataset): user item ratings Dataset object.

        """

        if "u" in self._dim.lower():
            return data.to_csr(centered_by="item")
        else:
            return data.to_csc(centered_by="user").T

    def _set_filepath(self) -> None:
        """Sets the filepath for the Matrix object in the destination directory."""
        filename = AdjustedCosineSimilarityMatrixFactory.__filenames[self._dim[0].lower()]
        self._filepath = os.path.join(self._destination, filename)


# ------------------------------------------------------------------------------------------------ #
#                                 PEARSON CORRELATION                                              #
# ------------------------------------------------------------------------------------------------ #
class PearsonSimilarityMatrixFactory(SimilarityMatrixFactory):

    __filenames = {
        "u": "pearson_correlation_similarity_user.pkl",
        "i": "pearson_correlation_similarity_item.pkl",
    }

    def __init__(
        self,
        name: str,
        desc: str,
        destination: str,
        dim: str,
        datasource: str = "movielens25m",
        force: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            desc=desc,
            destination=destination,
            dim=dim,
            datasource=datasource,
            force=force,
        )

    def _get_sparse_matrix(self, data: Matrix) -> Union[csc_matrix, csr_matrix]:
        """Obtains the sparse matrix for user or item dimensions.

        For user similarity, ratings are centered by item average ratings and
        vice-versa for item similarity.

        Args::
            data (Dataset): user item ratings Dataset object.

        """

        if "u" in self._dim.lower():
            return data.to_csr(centered_by="user")
        else:
            return data.to_csc(centered_by="item").T

    def _set_filepath(self) -> None:
        """Sets the filepath for the Matrix object in the destination directory."""
        filename = PearsonSimilarityMatrixFactory.__filenames[self._dim[0].lower()]
        self._filepath = os.path.join(self._destination, filename)
