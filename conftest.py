#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 08:08:04 am                                                #
# Modified   : Monday March 20th 2023 03:52:22 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest

from recsys.dataset.movielens import MovieLens
from recsys.services.io import IOService
from recsys.container import Recsys

# ------------------------------------------------------------------------------------------------ #
RATINGS_SMALL_FILEPATH = (
    "tests/testdata/operators/data_operators/ratings_user_random_sample_1pct.pkl"
)
RATINGS_FILEPATH = "tests/testdata/operators/data_operators/sampling/temporaralthreshold/ratings_random_temporal_sampling_1000.pkl"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataframe():
    return IOService.read(RATINGS_SMALL_FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def config_filepath():
    filepath = "tests/testdata/workflow/etl.yml"
    return filepath


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataset(dataframe):
    ds = MovieLens(
        name="test_dataset", desc="Test Dataset Sampled to 1000 Interactions", data=dataframe
    )
    return ds


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataset2(dataframe):
    ds = MovieLens(
        name="test_dataset", desc="Test Dataset 2 to Test Replacement Functionality", data=dataframe
    )
    return ds


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataset3(dataframe):
    ds = MovieLens(
        name="test_dataset_3",
        desc="Test Dataset 3 to Test Replacement Functionality",
        data=dataframe,
    )
    return ds


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def datasets(dataframe):
    datasets = []
    for i in range(1, 6):
        ds = MovieLens(
            name="test_dataset_" + str(i),
            desc="Test Dataset " + str(i) + " Description",
            data=dataframe,
        )
        datasets.append(ds)
    return datasets


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def container():
    container = Recsys()
    container.init_resources()
    container.wire(modules=["recsys.container", "recsys.asset.centre"])

    return container
