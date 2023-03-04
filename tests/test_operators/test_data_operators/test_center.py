#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_operators/test_data_operators/test_center.py                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 3rd 2023 02:17:33 am                                                   #
# Modified   : Friday March 3rd 2023 04:01:40 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from atelier.persistence.io import IOService

from recsys.operator.data.center import MeanCenter

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

SOURCE = "tests/testdata/operators/data_operators/ratings_sample_1pct.pkl"
SOURCE2 = "tests/testdata/operators/data_operators/ratings.csv"
DESTINATION = "tests/testdata/operators/data_operators/center/ratings_ctr.pkl"


@pytest.mark.ctr
class TestCenter:  # pragma: no cover
    # ============================================================================================ #
    def test_split(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        s = MeanCenter(source=SOURCE, destination=DESTINATION, force=True)
        s.execute()
        df = IOService.read(DESTINATION)
        assert "rating_ctr_user" in df.columns
        assert abs(df["rating"].values - df["rating_ctr_user"].values).all() > 0

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_force(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        s = MeanCenter(source=SOURCE2, destination=DESTINATION, force=False)
        s.execute()
        df = IOService.read(DESTINATION)
        assert "rating_ctr_user" in df.columns
        assert abs(df["rating"].values - df["rating_ctr_user"].values).all() > 0
        assert df.shape[0] < 10000000

        s = MeanCenter(
            by="movieId",
            column="rating_ctr_item",
            source=SOURCE2,
            destination=DESTINATION,
            force=True,
        )
        s.execute()
        df = IOService.read(DESTINATION)
        assert "rating_ctr_item" in df.columns
        assert abs(df["rating"].values - df["rating_ctr_item"].values).all() > 0
        assert df.shape[0] > 10000000

        logger.debug(df.head())

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
