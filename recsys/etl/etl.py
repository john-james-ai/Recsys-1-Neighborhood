#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/etl/etl.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 02:32:46 pm                                                 #
# Modified   : Saturday March 4th 2023 02:52:56 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import mlflow


# ------------------------------------------------------------------------------------------------ #
def extract(params):

    mlflow.run(
        uri=".",
        entry_point="etl",
        parameters=params,
        experiment_name="recsys_neighborhood_etl",
        experiment_id=1,
        run_name="recsys_extract",
    )


def transform(params):

    mlflow.run(
        uri=".",
        entry_point="transform",
        parameters=params,
        experiment_name="recsys_neighborhood_etl",
        experiment_id=1,
        run_name="recsys_transform",
    )


def load(params):

    run_name = "recsys_load_" + str(int(params["sample_size"] * 100)) + "_sample"

    mlflow.run(
        uri=".",
        entry_point="load",
        parameters=params,
        experiment_name="recsys_neighborhood_etl",
        experiment_id=1,
        run_name=run_name,
    )


def etl():

    extract_params = {
        "source": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "destination": "data/ext",
        "raw_dir": "data/raw",
        "member": "ratings.csv",
    }

    transform_params = {"source": "data/raw/ratings.csv", "destination": "data/raw/ratings.pkl"}

    load_prod_params = {
        "source": "data/raw/ratings.pkl",
        "destination": "data/prod/",
        "sample_size": 1,
        "train_size": 0.8,
        "train_filename": "train.pkl",
        "test_filename": "test.pkl",
    }

    load_dev_params = {
        "source": "data/raw/ratings.pkl",
        "destination": "data/dev/",
        "sample_size": 0.1,
        "train_size": 0.8,
        "train_filename": "train.pkl",
        "test_filename": "test.pkl",
    }
    with mlflow.start_run():
        extract(extract_params)
        transform(transform_params)
        load(load_prod_params)
        load(load_dev_params)


if __name__ == "__main__":
    etl()
