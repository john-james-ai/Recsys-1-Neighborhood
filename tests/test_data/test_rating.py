#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_data/test_rating.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 07:17:30 am                                                 #
# Modified   : Monday March 6th 2023 01:27:30 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd

from recsys.data.dataset import RatingsDataset

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

NAME = "test_ratings_dataset"
DESCRIPTION = "Test Ratings Dataset"


@pytest.mark.dataset
@pytest.mark.rating
class TestRating:  # pragma: no cover
    # ============================================================================================ #
    def test_properties(self, dataframe, caplog):
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
        ratings = RatingsDataset(name=NAME, description=DESCRIPTION, data=dataframe)
        assert isinstance(ratings.sparsity, float)
        assert isinstance(ratings.density, float)
        assert ratings.n_users < 1000000
        assert ratings.n_items < 100000
        assert ratings.utility_matrix_size > 10000
        assert len(ratings.users) < 1000000
        assert len(ratings.items) < 1000000
        assert ratings.user_item_ratio < 1
        assert ratings.item_user_ratio > 1
        assert isinstance(ratings.user_rating_frequency, pd.DataFrame)
        assert isinstance(ratings.item_rating_frequency, pd.DataFrame)
        assert isinstance(ratings.user_rating_frequency_distribution, pd.DataFrame)
        assert isinstance(ratings.item_rating_frequency_distribution, pd.DataFrame)

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
    def test_methods(self, dataframe, caplog):
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
        ratings = RatingsDataset(name=NAME, description=DESCRIPTION, data=dataframe)
        assert len(ratings.get_user_ratings(useridx=10)) > 2
        assert len(ratings.get_item_ratings(itemidx=10)) > 2
        assert len(ratings.get_users_rated_item(itemidx=10)) > 2
        assert len(ratings.get_items_rated_user(useridx=10)) > 2

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
    def test_center(self, dataframe, caplog):
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
        ratings = RatingsDataset(name=NAME, description=DESCRIPTION, data=dataframe)
        df = ratings.to_df()
        assert df.rating.gt(df.rating_ctr_user).all()
        assert df.rating.gt(df.rating_ctr_item).all()
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
    def test_reindex(self, dataframe, caplog):
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
        ratings = RatingsDataset(name=NAME, description=DESCRIPTION, data=dataframe)
        assert "useridx" in ratings.columns
        assert "itemidx" in ratings.columns
        assert ratings.ncols == 8

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
