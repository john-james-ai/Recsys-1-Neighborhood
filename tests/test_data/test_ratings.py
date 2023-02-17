#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_data/test_ratings.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 08:14:19 am                                                #
# Modified   : Friday February 17th 2023 04:29:18 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd
import numpy as np
from scipy import sparse


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

TRAIN_FILEPATH = "tests/data/train.pkl"
TEST_FILEPATH = "tests/data/test.pkl"
SAMPLE_FILEPATH = "tests/data/sample.pkl"


@pytest.mark.ratings
class TestRatingsDataset:  # pragma: no cover
    # ============================================================================================ #
    def test_summarize(self, ratings, caplog):
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
        summary = ratings.summarize()
        assert isinstance(summary, pd.DataFrame)
        assert summary.shape[0] == 13
        assert summary.shape[1] == 1
        logger.debug(summary)
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
    def test_info(self, ratings, caplog):
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
        ratings.info()
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
    def test_equality(self, ratings, ratings2, caplog):
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
        assert ratings == ratings
        assert ratings != ratings2
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
    def test_properties(self, ratings, caplog):
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
        assert isinstance(ratings.nrows, int)
        assert isinstance(ratings.ncols, int)
        assert isinstance(ratings.memory, float)
        assert isinstance(ratings.size, int)
        assert isinstance(ratings.matrix_size, int)
        assert isinstance(ratings.sparsity, float)
        assert isinstance(ratings.density, float)
        assert ratings.size > 0
        assert isinstance(ratings.n_users, int)
        assert isinstance(ratings.n_items, int)
        assert isinstance(ratings.users, np.ndarray)
        assert isinstance(ratings.items, np.ndarray)
        assert len(ratings.users) == ratings.n_users
        assert len(ratings.items) == ratings.n_items
        assert ratings.sparsity + ratings.density == 100

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
    def test_remap_id(self, ratings, caplog):
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
        df = ratings.as_dataframe()
        assert "itemidx" in df.columns
        assert "useridx" in df.columns

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
    def test_topn(self, ratings, caplog):
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
        top_users = ratings.top_n_users(n=10)
        assert isinstance(top_users, pd.DataFrame)
        assert top_users.shape[0 == 10]

        top_items = ratings.top_n_items(n=10)
        assert isinstance(top_items, pd.DataFrame)
        assert top_items.shape[0 == 10]

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
    def test_get_user_ratings(self, ratings, caplog):
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
        user = 8089
        ru = ratings.get_user_ratings(user)
        assert isinstance(ru, pd.DataFrame)
        assert ru.shape[0] == 15

        ru = ratings.get_user_ratings(user, mean_centered="user")
        assert isinstance(ru, pd.DataFrame)
        assert ru.shape[0] == 15
        assert ru["rating"].min() < 0

        ru = ratings.get_user_ratings(user, mean_centered="item")
        assert isinstance(ru, pd.DataFrame)
        assert ru.shape[0] == 15
        assert ru["rating"].min() < 0

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
    def test_get_items_ratings(self, ratings, caplog):
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
        item = 212
        ri = ratings.get_item_ratings(item)
        assert isinstance(ri, pd.DataFrame)
        assert ri.shape[0] == 69
        assert ri["rating"].min() >= 0
        assert ri["rating"].max() <= 5

        ri = ratings.get_item_ratings(item, mean_centered="user")
        assert isinstance(ri, pd.DataFrame)
        assert ri.shape[0] == 69
        assert ri["rating"].min() < 0

        ri = ratings.get_item_ratings(item, mean_centered="item")
        assert isinstance(ri, pd.DataFrame)
        assert ri.shape[0] == 69
        assert ri["rating"].min() < 0
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
    def test_user_item_rating(self, ratings, caplog):
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
        user = 8089
        item = 862
        rui1 = ratings.get_user_item_rating(user, item)
        assert isinstance(rui1, float)
        rui2 = ratings.get_user_item_rating(user, item, mean_centered="user")
        assert isinstance(rui2, float)
        rui3 = ratings.get_user_item_rating(user, item, mean_centered="item")
        assert isinstance(rui3, float)
        assert rui1 != rui2
        assert rui3 != rui2
        assert rui1 != rui3

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
    def test_ave_user_rating(self, ratings, caplog):
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
        u = 8089
        rbar = ratings.get_ave_user_ratings(useridx=u)
        assert isinstance(rbar, float)
        assert np.isclose(rbar, 3.233333, rtol=1e-2)

        rbar = ratings.get_ave_user_ratings()
        assert isinstance(rbar, pd.DataFrame)
        assert np.isclose(rbar[rbar["useridx"] == u]["rbar"].values[0], 3.233333, rtol=1e-2)

        with pytest.raises(ValueError):
            u = 98765
            ratings.get_ave_user_ratings(useridx=u)

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
    def test_ave_item_rating(self, ratings, caplog):
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
        i = 212
        rbar = ratings.get_ave_item_ratings(itemidx=i)
        assert isinstance(rbar, float)
        assert np.isclose(rbar, 4.05, rtol=1e-2)

        rbar = ratings.get_ave_item_ratings()
        assert isinstance(rbar, pd.DataFrame)
        assert np.isclose(rbar[rbar["itemidx"] == i]["rbar"].values[0], 4.07, rtol=1e-2)

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
    def test_as_dataframe(self, ratings, caplog):
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
        df = ratings.as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == ratings.nrows
        assert df.shape[1] == ratings.ncols
        assert df.useridx.nunique() == ratings.n_users
        assert df.itemidx.nunique() == ratings.n_items

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
    def test_as_csr(self, ratings, caplog):
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
        # Raw ratings user
        csr = ratings.as_csr()
        shape = csr.shape
        assert shape[0] > shape[1]
        assert isinstance(csr, sparse.csr_matrix)
        assert csr.min() >= 0

        # Raw ratings item
        csr = ratings.as_csr(axis=1)
        shape = csr.shape
        assert shape[0] < shape[1]
        assert isinstance(csr, sparse.csr_matrix)
        assert csr.min() >= 0

        # user mean centered ratings
        csr = ratings.as_csr(mean_centered="user")
        assert isinstance(csr, sparse.csr_matrix)
        assert csr.min() < 0

        # item mean centered ratings
        csr = ratings.as_csr(mean_centered="item")
        assert isinstance(csr, sparse.csr_matrix)
        assert csr.min() < 0

        with pytest.raises(ValueError):
            ratings.as_csr(mean_centered="29")

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
    def test_as_csc(self, ratings, caplog):
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
        # Raw ratings
        csc = ratings.as_csc()
        shape = csc.shape
        assert shape[0] > shape[1]
        assert isinstance(csc, sparse.csc_matrix)
        assert csc.min() >= 0

        # Raw ratings items
        csc = ratings.as_csc(axis=1)
        shape = csc.shape
        assert shape[0] < shape[1]
        assert isinstance(csc, sparse.csc_matrix)
        assert csc.min() >= 0

        # user mean centered ratings
        csc = ratings.as_csc(mean_centered="user")
        assert isinstance(csc, sparse.csc_matrix)
        assert csc.min() < 0

        # item mean centered ratings
        csc = ratings.as_csc(mean_centered="item")
        assert isinstance(csc, sparse.csc_matrix)
        assert csc.min() < 0

        with pytest.raises(ValueError):
            ratings.as_csc(mean_centered="29")

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
    def test_center(self, ratings, caplog):
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
        user = 72315
        item = 162644
        ratings.center()
        df = ratings.as_dataframe()
        assert "rating_ci" in df.columns
        assert "rating_cu" in df.columns
        assert np.greater(df.rating.values, df.rating_ci.values).all()
        assert np.greater(df.rating.values, df.rating_cu.values).all()
        assert np.greater(
            df[(df["userId"] == user) & (df["movieId"] == item)]["rating"].values,
            df[(df["userId"] == user) & (df["movieId"] == item)]["rating_cu"].values,
        ).all()
        assert np.greater(
            df[(df["userId"] == user) & (df["movieId"] == item)]["rating"].values,
            df[(df["userId"] == user) & (df["movieId"] == item)]["rating_ci"].values,
        ).all()
        df1 = df[(df["userId"] == user) & (df["movieId"] == item)]
        df2 = df[(df["userId"] == user) & (df["movieId"] == item)]
        logger.debug(f"{df1}, {df2}")
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
