#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/similarity.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 2nd 2023 03:44:46 pm                                              #
# Modified   : Thursday February 2nd 2023 09:08:46 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Similarity Matrix Module"""
import numpy as np
from tqdm import tqdm
from itertools import combinations

from recsys.data.rating import RatingsDataset
from recsys.neighborhood.base import Matrix, Similarity

# ------------------------------------------------------------------------------------------------ #
class Cosign(Similarity):
    def __init__(self) -> None:
        super().__init__()

    def compute_user_similarity(self, ratings: RatingsDataset) -> Matrix:
        Iuv = self._compute_user_pairs_per_item()

        for uv, uv_ratings in tqdm(I.items()):
            u, v = from_key(uv)
            ru = uv_ratings[uv_ratings["userId"] == u].sort_values(by="movieId")["rating"].values
            rv = uv_ratings[uv_ratings["userId"] == v].sort_values(by="movieId")["rating"].values
            S[uv] = ru.dot(rv) / (N[u] * N[v])
