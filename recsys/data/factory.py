#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/factory.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 05:01:09 am                                                #
# Modified   : Wednesday March 1st 2023 06:12:04 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Similarity Matrix Module"""
from abc import ABC, abstractmethod

from recsys.data.dataset import Dataset
from recsys.operator.similarity.base import SimilarityMeasure
from recsys.data.similarity import SimilarityMatrix


# ------------------------------------------------------------------------------------------------ #
class AbstractSimilarityFactory(ABC):
    """Creates a similarity matrix.

    Similarity may be measured between items or users. The variants are implemented by the
    subclasses.
    """

    def __init__(self, dataset: Dataset, measure: SimilarityMeasure) -> None:
        self._dataset = dataset
        self._measure = measure

    @abstractmethod
    def compute_corating_matrix(self) -> dict:
        """Returns the list of dictionaries containing the corated items or users."""

    @abstractmethod
    def compute_similarity_matrix(self) -> SimilarityMatrix:
        """Creates similarity matrix using the corated item / userse computed above."""


# ------------------------------------------------------------------------------------------------ #
class UserSimilarityMatrixFactory(AbstractSimilarityFactory):
    """Creates user similarity matrix."""

    def __init__(self, dataset: Dataset, measure: SimilarityMeasure) -> None:
        super().__init__(dataset=dataset, measure=measure)

    def compute_corating_matrix(self) -> dict:
        """Obtains the items corated by pairs of users"""

    def compute_similarity_matrix(self) -> SimilarityMatrix:
        """Computes similarity for each pair of users"""


# ------------------------------------------------------------------------------------------------ #
class ItemSimilarityMatrixFactory(AbstractSimilarityFactory):
    """Creates item similarity matrix."""

    def __init__(self, dataset: Dataset, measure=SimilarityMeasure) -> None:
        super().__init__(dataset=dataset, measure=measure)

    def compute_corating_matrix(self) -> dict:
        """Obtains the users who have rated both items in a pair of items"""

    def compute_similarity_matrix(self) -> SimilarityMatrix:
        """Computes similarity for each pair of items"""
