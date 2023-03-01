#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/similarity/base.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 05:05:32 am                                                #
# Modified   : Wednesday March 1st 2023 06:02:45 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import abstractmethod

from recsys.operator.base import Operator
from recsys.data.dataset import Dataset
from recsys.data.similarity import SimilarityDataset


# ------------------------------------------------------------------------------------------------ #
class SimilarityMeasure(Operator):
    """Base class for Similarity Methods"""

    @abstractmethod
    def execute(self, data: Dataset) -> SimilarityDataset:
        """Computes the similarity measure"""
