#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_operators/test_data_operators/test_split.py                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 3rd 2023 02:17:33 am                                                   #
# Modified   : Saturday March 18th 2023 08:41:29 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging

from recsys import IOService

from recsys.data_selection.split_builder import TemporalTrainTestSplit

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

SOURCE = "tests/testdata/operators/data_operators/ratings.csv"
SOURCE2 = "tests/testdata/operators/data_operators/ratings_sample_1pct.pkl"
DESTINATION = "tests/testdata/operators/data_operators/split/"
TRAIN_FILEPATH = os.path.join(DESTINATION, "train.pkl")
TEST_FILEPATH = os.path.join(DESTINATION, "test.pkl")


@pytest.mark.split
class TestSplit:  # pragma: no cover
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
        s = TemporalTrainTestSplit(source=SOURCE, destination=DESTINATION, force=True)
        s.__call__()
        assert os.path.exists(TRAIN_FILEPATH)
        assert os.path.exists(TEST_FILEPATH)
        df1 = IOService.read(TRAIN_FILEPATH)
        df2 = IOService.read(TEST_FILEPATH)
        assert df1.shape[0] > df2.shape[0] * 3

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
        s = TemporalTrainTestSplit(
            train_size=0.5, source=SOURCE, destination=DESTINATION, force=False
        )
        s.__call__()
        df1 = IOService.read(TRAIN_FILEPATH)
        df2 = IOService.read(TEST_FILEPATH)
        assert df1.shape[0] > df2.shape[0] * 3

        s = TemporalTrainTestSplit(
            train_size=0.5, source=SOURCE2, destination=DESTINATION, force=True
        )
        s.__call__()
        df1 = IOService.read(TRAIN_FILEPATH)
        df2 = IOService.read(TEST_FILEPATH)
        assert df1.shape[0] < df2.shape[0] * 4

        # with pytest.raises(KeyError):
        #     TemporalTrainTestSplit(
        #         train_size=0.5,
        #         source=SOURCE,
        #         destination=DESTINATION,
        #         timestamp_var="xyz",
        #         force=True,
        #     )

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
