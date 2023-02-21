#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/neighborhood/offline.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 3rd 2023 10:23:59 pm                                                #
# Modified   : Monday February 20th 2023 09:59:31 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from tqdm import tqdm
import pandas as pd
from itertools import combinations
from multiprocessing import Pool

from recsys.dal.rating import RatingsDataset
from recsys.neighborhood.matrix import SimilarityMatrix, UserSimilarityMatrix

FILEPATH = "data/dev/ratings_0.5_pct.pkl"


# ------------------------------------------------------------------------------------------------ #
def to_key(uv: tuple) -> str:
    return str(uv[0]) + "_" + str(uv[1])


def from_key(uv: str) -> tuple:
    return tuple([int(x) for x in uv.split("_")])


# ------------------------------------------------------------------------------------------------ #
def create_dataset(filepath: str) -> RatingsDataset:
    return RatingsDataset(filepath=filepath)


# ------------------------------------------------------------------------------------------------ #
def create_user_pairs(ratings: RatingsDataset):
    UV = {}

    for item in tqdm(ratings.items):
        item_ratings = ratings.get_item_ratings(item=item)
        for uv_pair in combinations(item_ratings["userId"].values, 2):
            uv_key = to_key(uv_pair)
            if UV.get(uv_key, None) is not None:
                UV[uv_key].append(item)
            else:
                UV[uv_key] = [item]
    return UV


# ------------------------------------------------------------------------------------------------ #
def compute_similarity(uv: str, items: list, ratings: RatingsDataset) -> SimilarityMatrix:
    u, v = from_key(uv)
    ru = ratings.get_users_items_ratings(items=items, users=[u])
    rv = ratings.get_users_items_ratings(items=items, users=[v])
    l2u = ratings.get_user_rating_norms(user=u)
    l2v = ratings.get_user_rating_norms(user=v)
    sim = ru.dot(rv) / (l2u * l2v)
    sim = {"u": u, "v": v, "sim": sim}
    return sim


# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    ratings = create_dataset(filepath=FILEPATH)
    uv_pairs = create_user_pairs(ratings=ratings)

    with Pool(processes=12) as pool, tqdm(total=len(uv_pairs)) as pbar:
        similarity_data = [
            pool.apply_async(
                compute_similarity,
                (
                    uv,
                    items,
                    ratings,
                ),
                callback=lambda _: pbar.update(1),
            )
            for uv, items in uv_pairs.items()
        ]
    similarity_data = pd.DataFrame(similarity_data)
    similarity_matrix = UserSimilarityMatrix(name="user_similarity_matrix", data=similarity_data)
    similarity_matrix.save()
