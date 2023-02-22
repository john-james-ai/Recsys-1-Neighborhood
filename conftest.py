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
# Modified   : Wednesday February 22nd 2023 01:32:12 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest

from recsys.persistence.fio import IOService
from recsys.container import Recsys
from recsys.dal.dto import DatasetDTO


# ------------------------------------------------------------------------------------------------ #
RATINGS_FILEPATH = "tests/data/train.pkl"
TEST_RATINGS_FILEPATH = "tests/data/test_ratings.csv"
TEST_DATAFRAME_FILEPATH = "tests/data/dataframe.csv"
SQLALCHEMY = "sqlite:///tests/data/test.db"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def dataframe():
    return IOService.read(filepath=TEST_DATAFRAME_FILEPATH)


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def container():
    container = Recsys()
    container.init_resources()
    container.wire(
        modules=["recsys.container"],
    )

    return container


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session")
def dtos():
    dtos = []
    for i in range(1, 6):
        dto = DatasetDTO(
            id=i,
            name="dto_" + str(i),
            type="test_dto_type",
            description="dto_description_" + str(i),
            workspace="dev",
            filepath="some_test_filepath",
            stage="stage_" + str(i * 10),
            rows=i * 22,
            cols=i + 14,
            n_users=i * 1000,
            n_items=i * 2000,
            size=i + 8772,
            matrix_size=i + 5362,
            memory_mb=i * 46,
            cost=i * 264,
            sparsity=85,
            density=15,
        )
        dtos.append(dto)
    return dtos
