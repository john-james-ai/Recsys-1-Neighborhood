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
# Modified   : Saturday February 4th 2023 12:11:49 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Similarity Matrix Module"""
import numpy as np
import pandas as pd

from tqdm import tqdm

from recsys.data.rating import RatingsDataset
from recsys.neighborhood.base import SimilarityMetric
from recsys.neighborhood.matrix import UserSimilarityMatrix, ItemSimilarityMatrix


# ------------------------------------------------------------------------------------------------ #
class CosignSimilarity(SimilarityMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute_user_similarity(self, ratings: RatingsDataset) -> UserSimilarityMatrix:

        S = {}

        Iuv = self._extract_common_items()

        for uv, uv_ratings in tqdm(Iuv.items()):
            u, v = self._from_key(uv)
            rui = uv_ratings[uv_ratings["userId"] == u].sort_values(by="movieId")["rating"].values
            rvi = uv_ratings[uv_ratings["userId"] == v].sort_values(by="movieId")["rating"].values
            l2u = ratings.get_user_rating_norms(user=u)
            l2v = ratings.get_user_rating_norms(user=v)
            S[uv] = rui.dot(rvi) / (l2u * l2v)
        similarity = pd.DataFrame.from_dict(
            data=S, orient="index", columns=["similarity"]
        ).reset_index()

        similarity[["u", "v"]] = similarity["index"].str.split("_", 1, expand=True)
        self._user_similarity = UserSimilarityMatrix(
            name="cosign_user_similarity_matrix", data=similarity
        )

    def compute_item_similarity(self, ratings: RatingsDataset) -> ItemSimilarityMatrix:

        S = {}

        Uij = self._compute_item_pairs_per_item()

        for ij, ij_ratings in tqdm(Uij.items()):
            i, j = self._from_key(ij)
            riu = ij_ratings[ij_ratings["movieId"] == i].sort_values(by="userId")["rating"].values
            rju = ij_ratings[ij_ratings["movieId"] == j].sort_values(by="userId")["rating"].values
            l2i = np.sqrt(np.sum(ratings.get_item_ratings(user=i) ** 2))
            l2j = np.sqrt(np.sum(ratings.get_item_ratings(user=j) ** 2))
            S[ij] = riu.dot(rju) / (l2i * l2j)
        similarity = pd.DataFrame.from_dict(
            data=S, orient="index", columns=["similarity"]
        ).reset_index()

        similarity[["i", "j"]] = similarity["index"].str.split("_", 1, expand=True)
        self._item_similarity = ItemSimilarityMatrix(
            name="cosign_item_similarity_matrix", data=similarity
        )
