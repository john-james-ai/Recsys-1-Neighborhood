#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 08:08:04 am                                                #
# Modified   : Tuesday February 28th 2023 05:56:48 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest
from types import SimpleNamespace

from recsys.container import Recsys
from recsys.persistence.io import IOService
from recsys.dataset.rating import RatingsDataset
from recsys.persistence.odb import ObjectDB, CacheDB  # noqa F401
from recsys.persistence.cache import CacheObject

# ------------------------------------------------------------------------------------------------ #
#                                         FILEPATHS                                                #
# ------------------------------------------------------------------------------------------------ #
#                                         TEST FILE                                                #
ZIPFILE_URL = "https://stats.govt.nz/assets/Uploads/International-trade/International-trade-September-2022-quarter/Download-data/international-trade-september-2022-quarter-csv.zip"
ZIPFILE_DOWNLOAD = "tests/data/download/international-trade-september-2022-quarter-csv.zip"
ZIPFILE_EXTRACT = "tests/data/extract/"
RATINGS_CSV = "tests/data/test_ratings.csv"
RATINGS_PKL = "tests/data/ratings_1_pct.pkl"
TEST_DATABASE = "tests/data/database/test.odb"
TEST_CACHE = "tests/data/cache/test.odb"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def files():
    files = {
        "zipfile_url": ZIPFILE_URL,
        "zipfile_download": ZIPFILE_DOWNLOAD,
        "zipfile_extract": ZIPFILE_EXTRACT,
        "ratings_csv": RATINGS_CSV,
        "ratings_pkl": RATINGS_PKL,
    }
    files = SimpleNamespace(**files)
    return files


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def config():
    FILEPATH = "tests/data/test_workflow_config.yml"
    return IOService.read(FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def ratings():
    return IOService.read(RATINGS_PKL)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def dataset(ratings):
    return RatingsDataset(name="test_dataset", description="Test Ratings Dataset", data=ratings)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def datasets(ratings):
    ds = []
    for i in range(1, 6):
        name = "test_dataset_" + str(i)
        desc = "Test Ratings Dataset " + str(i)
        r = RatingsDataset(name=name, description=desc, data=ratings)
        ds.append(r)
    return ds


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def odb():
    return ObjectDB(filepath=TEST_DATABASE)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def cache():
    return CacheDB(filepath=TEST_CACHE)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def cache_objects(ratings):
    objects = []
    timedeltas = [-7, 0, 7]
    for i, delta in enumerate(timedeltas):
        obj = CacheObject(key=str(i), ttl=delta, content=ratings)
        objects.append(obj)
    return objects


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def container():
    container = Recsys()
    container.init_resources()
    container.wire(modules=["recsys.container", "recsys.persistence.cache"])

    return container
