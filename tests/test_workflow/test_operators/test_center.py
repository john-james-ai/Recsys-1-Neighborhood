#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_workflow/test_operators/test_center.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 03:12:23 am                                             #
# Modified   : Tuesday February 28th 2023 11:55:30 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging
import shutil
import numpy as np

from recsys.operator.data.center import MeanCenter
from recsys.persistence.io import IOService

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.operator
@pytest.mark.transformer
@pytest.mark.centerer
class TestRatingCenterer:  # pragma: no cover

    DESTINATION = "tests/results/operators/transformer/ratings.pkl"

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
        shutil.rmtree(os.path.dirname(TestRatingCenterer.DESTINATION), ignore_errors=True)
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
    def test_center(self, ratings, files, caplog):
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
        c = MeanCenter(source=files.ratings_pkl, destination=TestRatingCenterer.DESTINATION)
        c.execute()
        assert os.path.exists(TestRatingCenterer.DESTINATION)

        ratings = IOService.read(TestRatingCenterer.DESTINATION)
        assert "rating_cbu" in ratings.columns
        assert "rating_cbi" in ratings.columns

        USERID = 48653
        ITEMID = 8633
        ITEM_CENTERED_RATINGS = [0, -0.5, 0, 0.5]
        USER_CENTERED_RATINGS = [0.1, -0.4, 0.1, 0.1, 0.1]

        ratings_cbu = ratings[ratings["userId"] == USERID]["rating_cbu"].values
        ratings_cbi = ratings[ratings["movieId"] == ITEMID]["rating_cbi"].values

        logger.debug(f"\nRatings centered by item: \n{ratings_cbi}")
        logger.debug(f"\nRatings centered by user: \n{ratings_cbu}")

        assert np.isclose(ratings_cbi, ITEM_CENTERED_RATINGS).all()
        assert np.isclose(ratings_cbu, USER_CENTERED_RATINGS).all()

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
        c = MeanCenter(
            source=files.ratings_pkl,
            destination=TestRatingCenterer.DESTINATION,
            uservar="incorrect",
            force=True,
        )
        with pytest.raises(ValueError):
            c.execute()

        c = MeanCenter(
            source=files.ratings_pkl,
            destination=TestRatingCenterer.DESTINATION,
            itemvar="incorrect",
            force=True,
        )
        with pytest.raises(ValueError):
            c.execute()

        c = MeanCenter(
            source=files.ratings_pkl,
            destination=TestRatingCenterer.DESTINATION,
            rating_var="incorrect",
            force=True,
        )
        with pytest.raises(ValueError):
            c.execute()

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
