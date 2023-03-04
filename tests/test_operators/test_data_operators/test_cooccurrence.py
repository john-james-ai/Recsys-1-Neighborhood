#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_operators/test_data_operators/test_cooccurrence.py                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 3rd 2023 02:17:33 am                                                   #
# Modified   : Friday March 3rd 2023 06:58:27 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import itertools

from atelier.persistence.io import IOService

from recsys.operator.data.cooccurrence import UserCooccurrenceIndex, ItemCooccurrenceIndex

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

SOURCE1 = "tests/testdata/operators/data_operators/ratings_user_random_sample_1pct.pkl"
SOURCE2 = "tests/testdata/operators/data_operators/ratings_user_random_sample_10pct.pkl"
DESTINATION1 = "tests/testdata/operators/data_operators/cooccurrence/user_ratings_ctr_10pct.pkl"
DESTINATION2 = "tests/testdata/operators/data_operators/cooccurrence/item_ratings_ctr_10pct.pkl"


@pytest.mark.pairs
class TestUserCooccurrence:  # pragma: no cover
    # ============================================================================================ #
    def test_cooccurrence(self, caplog):
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
        s = UserCooccurrenceIndex(source=SOURCE1, destination=DESTINATION1, force=True)
        s.execute()
        d2 = IOService.read(DESTINATION1)
        assert isinstance(d2, dict)
        logger.debug(dict(itertools.islice(d2.items(), 4)))

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
        s = UserCooccurrenceIndex(source=SOURCE1, destination=DESTINATION1, force=False)
        s.execute()
        d2 = IOService.read(DESTINATION1)
        assert isinstance(d2, dict)

        s = UserCooccurrenceIndex(source=SOURCE2, destination=DESTINATION1, force=True)
        s.execute()
        d2 = IOService.read(DESTINATION1)
        assert isinstance(d2, dict)

        logger.debug(dict(itertools.islice(d2.items(), 4)))

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


# @pytest.mark.pairs
class TestItemCooccurrence:  # pragma: no cover
    # ============================================================================================ #
    def test_cooccurrence(self, caplog):
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
        s = ItemCooccurrenceIndex(source=SOURCE1, destination=DESTINATION2, force=True)
        s.execute()
        d2 = IOService.read(DESTINATION2)
        assert isinstance(d2, dict)

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
        s = ItemCooccurrenceIndex(source=SOURCE1, destination=DESTINATION2, force=False)
        s.execute()
        d2 = IOService.read(DESTINATION2)
        assert isinstance(d2, dict)

        s = ItemCooccurrenceIndex(source=SOURCE2, destination=DESTINATION2, force=True)
        s.execute()
        d2 = IOService.read(DESTINATION2)
        assert isinstance(d2, dict)

        logger.debug(dict(itertools.islice(d2.items(), 2)))

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
