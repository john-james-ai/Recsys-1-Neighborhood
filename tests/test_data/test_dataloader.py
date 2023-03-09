#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /tests/test_data/test_dataloader.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 6th 2023 01:09:22 am                                                   #
# Modified   : Monday March 6th 2023 02:37:03 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from recsys.data.dataloader import TabularDataLoader


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

FILEPATH = "tests/testdata/operators/data_operators/ratings_user_random_sample_10pct.pkl"


@pytest.mark.loader
class TestDataLoader:  # pragma: no cover
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
        dl = TabularDataLoader(filepath=FILEPATH)

        ds = next(dl)
        logger.debug(ds.head())

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
