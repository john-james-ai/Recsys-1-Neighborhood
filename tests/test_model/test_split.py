#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_model/test_split.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 4th 2023 01:31:35 am                                              #
# Modified   : Saturday February 4th 2023 01:46:24 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import numpy as np
import inspect
from datetime import datetime
import pytest
import logging

from recsys.model.split import TemporalSplitter

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.split
class TestSplit:  # pragma: no cover
    # ============================================================================================ #
    def test_split(self, ratings_df, caplog):
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
        train_size = 0.8
        train_filepath = "tests/data/train.pkl"
        test_filepath = "tests/data/test.pkl"
        split_var = "timestamp"
        splitter = TemporalSplitter()
        data = splitter.split(
            data=ratings_df,
            split_var=split_var,
            train_size=train_size,
            train_filepath=train_filepath,
            test_filepath=test_filepath,
        )
        assert os.path.exists(train_filepath)
        assert os.path.exists(test_filepath)
        assert len(data["train"]) == len(ratings_df) * train_size
        assert np.isclose(len(data["test"]), len(ratings_df) * (1 - train_size))

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
    def test_split_error(self, ratings_df, caplog):
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
        train_size = 0.8
        train_filepath = "tests/data/train.pkl"
        test_filepath = "tests/data/test.pkl"
        split_var = "non-existing"
        splitter = TemporalSplitter()
        with pytest.raises(ValueError):
            splitter.split(
                data=ratings_df,
                split_var=split_var,
                train_size=train_size,
                train_filepath=train_filepath,
                test_filepath=test_filepath,
            )
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
