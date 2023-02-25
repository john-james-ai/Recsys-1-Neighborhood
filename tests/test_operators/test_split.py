#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_operators/test_split.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 01:34:13 am                                             #
# Modified   : Saturday February 25th 2023 09:31:52 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
import math
from datetime import datetime
import pytest
import logging
import shutil

from recsys.data.split import TemporalTrainTestSplit
from recsys.io.service import IOService

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.operator
@pytest.mark.split
class TestSplit:  # pragma: no cover

    SOURCE = "tests/data/ratings_10_pct.pkl"
    DESTINATION = "tests/results/operators/split/"
    TRAIN = "train.pkl"
    TEST = "test.pkl"
    TRAIN_FILEPATH = os.path.join(DESTINATION, TRAIN)
    TEST_FILEPATH = os.path.join(DESTINATION, TEST)

    # ============================================================================================ #
    def test_setup(self, caplog):
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
        shutil.rmtree(TestSplit.DESTINATION, ignore_errors=True)
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
    def test_split(self, files, caplog):
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
        tts = TemporalTrainTestSplit(
            source=files.ratings_pkl,
            destination=TestSplit.DESTINATION,
            train_filename=TestSplit.TRAIN,
            test_filename=TestSplit.TEST,
        )
        tts.execute()
        assert os.path.exists(TestSplit.TRAIN_FILEPATH)
        assert os.path.exists(TestSplit.TEST_FILEPATH)
        assert tts.source == files.ratings_pkl
        assert tts.destination == TestSplit.DESTINATION

        orig = IOService.read(files.ratings_pkl)
        train = IOService.read(TestSplit.TRAIN_FILEPATH)
        test = IOService.read(TestSplit.TEST_FILEPATH)

        orig_shape = orig.shape[0]
        tgt_train_shape = orig_shape * 0.8
        tgt_test_shape = orig_shape * 0.2

        assert math.isclose(train.shape[0], tgt_train_shape, rel_tol=1e-05)
        assert math.isclose(test.shape[0], tgt_test_shape, rel_tol=1e-05)

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

    # ============================================================================================ #
    def test_force(self, files, caplog):
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
        tts = TemporalTrainTestSplit(
            source=files.ratings_pkl,
            destination=TestSplit.DESTINATION,
            train_filename=TestSplit.TRAIN,
            test_filename=TestSplit.TEST,
            force=True,
        )
        tts.execute()

        assert tts.status == "success"
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

    # ============================================================================================ #
    def test_errors(self, files, caplog):
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
        shutil.rmtree(TestSplit.DESTINATION, ignore_errors=True)
        tts = TemporalTrainTestSplit(
            source=files.ratings_pkl,
            destination=TestSplit.DESTINATION,
            train_filename=TestSplit.TRAIN,
            test_filename=TestSplit.TEST,
            timestamp_var="invalid",
        )
        with pytest.raises(ValueError):
            tts.execute()

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
