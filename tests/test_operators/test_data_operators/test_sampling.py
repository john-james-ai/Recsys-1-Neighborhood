#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_operators/test_data_operators/test_sampling.py                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 3rd 2023 02:17:33 am                                                   #
# Modified   : Friday March 3rd 2023 04:07:23 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from atelier.persistence.io import IOService

from recsys.operator.data.sampling import UserRandomSampling, UserStratifiedRandomSampling

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

SOURCE = "tests/testdata/operators/data_operators/ratings.csv"
SOURCE2 = "tests/testdata/operators/data_operators/ratings_sample_10pct.pkl"
USERVAR = "userId"
ITEMVAR = "movieId"
DESTINATION1 = "tests/testdata/operators/data_operators/ratings_user_random_sample_10pct.pkl"
DESTINATION2 = (
    "tests/testdata/operators/data_operators/ratings_user_stratified_random_sample_1pct.pkl"
)


@pytest.mark.sample
class TestUserRandomSampling:  # pragma: no cover
    # ============================================================================================ #
    def test_sampling(self, caplog):
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
        s = UserRandomSampling(
            frac=0.1, uservar=USERVAR, itemvar=ITEMVAR, source=SOURCE, destination=DESTINATION1
        )
        s.execute()
        df1 = IOService.read(SOURCE)
        df2 = IOService.read(DESTINATION1)
        assert df1.shape[0] > df2.shape[0] * 9

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
        s = UserRandomSampling(
            frac=0.1,
            uservar=USERVAR,
            itemvar=ITEMVAR,
            source=SOURCE,
            destination=DESTINATION1,
            force=False,
        )
        s.execute()
        df1 = IOService.read(SOURCE)
        df2 = IOService.read(DESTINATION1)
        assert df1.shape[0] > df2.shape[0] * 9
        s = UserRandomSampling(
            frac=0.1,
            uservar=USERVAR,
            itemvar=ITEMVAR,
            source=SOURCE,
            destination=DESTINATION1,
            force=True,
        )
        s.execute()
        df1 = IOService.read(SOURCE)
        df2 = IOService.read(DESTINATION1)
        assert df1.shape[0] > df2.shape[0] * 9
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


@pytest.mark.sample
@pytest.mark.skip()
class TestUserStratifiedRandomSampling:  # pragma: no cover
    # ============================================================================================ #
    def test_sampling(self, caplog):
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
        s = UserStratifiedRandomSampling(
            frac=0.01, uservar=USERVAR, itemvar=ITEMVAR, source=SOURCE, destination=DESTINATION2
        )
        s.execute()
        df1 = IOService.read(SOURCE)
        df2 = IOService.read(DESTINATION2)
        assert df1.shape[0] > df2.shape[0] * 90

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
        s = UserStratifiedRandomSampling(
            frac=0.1,
            uservar=USERVAR,
            itemvar=ITEMVAR,
            source=SOURCE,
            destination=DESTINATION2,
            force=False,
        )
        s.execute()
        df1 = IOService.read(SOURCE)
        df2 = IOService.read(DESTINATION2)
        assert df1.shape[0] > df2.shape[0] * 90
        s = UserStratifiedRandomSampling(
            frac=0.1,
            uservar=USERVAR,
            itemvar=ITEMVAR,
            source=SOURCE2,
            destination=DESTINATION2,
            force=True,
        )
        s.execute()
        df1 = IOService.read(SOURCE2)
        df2 = IOService.read(DESTINATION2)
        assert df1.shape[0] < df2.shape[0] * 90
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
