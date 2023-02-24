#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/matrix.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 08:37:49 pm                                                #
# Modified   : Wednesday February 22nd 2023 11:05:31 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Matrix Module"""
from scipy.sparse import csr_matrix, csc_matrix
from typing import Union

from recsys.persistence.rating import RatingsDataset
from recsys.data.base import Matrix


# ------------------------------------------------------------------------------------------------ #
class SimilarityMatrix(Matrix):
    def __init__(
        self,
        method: str,
        matrix: Union[csr_matrix, csc_matrix],
        ratings: RatingsDataset,
        user: bool = True,
    ) -> None:
        super().__init__(matrix=matrix, ratings=ratings, user=user)
        self._method = method
        self._set_name()

    def get_similarity(self, a: int, b: int) -> float:
        key = (a, b) if a < b else (b, a)
        return self._matrix[key]

    def _set_name(self) -> None:
        matrix_type = "user" if self._user else "item"
        self._name = (
            self._method + "_" + matrix_type + "_similarity_" + self._mode + "_" + self._dataset
        )
