#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 08:08:04 am                                                #
# Modified   : Thursday March 9th 2023 06:43:48 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest

from recsys.data.dataset import Dataset
from recsys.data.matrix import Matrix
from recsys.services.io import IOService
from recsys.container import Recsys

# ------------------------------------------------------------------------------------------------ #
RATINGS_FILEPATH = "tests/testdata/operators/data_operators/sampling/temporaralthreshold/ratings_random_temporal_sampling_1000.pkl"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataframe():
    return IOService.read(RATINGS_FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def config_filepath():
    filepath = "tests/testdata/workflow/etl.yml"
    return filepath


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataset(dataframe):
    ds = Dataset(
        name="test_dataset", description="Test Dataset Sampled to 1000 Interactions", data=dataframe
    )
    return ds


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def interaction(dataset):
    out = Matrix(
        name="test_interaction_matrix", description="Test Interaction Matrix", data=dataset.as_coo()
    )
    return out


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def container():
    container = Recsys()
    container.init_resources()
    container.wire(modules=["recsys.container"])

    return container
