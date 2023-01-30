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
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 08:08:04 am                                                #
# Modified   : Monday January 30th 2023 06:13:57 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest

from recsys.data.rating import RatingsDataset
from recsys.io.file import IOService
from recsys.container import Recsys

# ------------------------------------------------------------------------------------------------ #
RATINGS_FILEPATH = "tests/data/rating.pkl"
TEST_RATINGS_FILEPATH = "tests/data/test_ratings.csv"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def ratings():
    return RatingsDataset(filepath=RATINGS_FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def test_ratings():
    return IOService.read(TEST_RATINGS_FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="class", autouse=True)
def container():
    container = Recsys()
    container.init_resources()
    return container
