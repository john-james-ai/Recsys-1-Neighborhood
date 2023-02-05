#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_neighborhood/test_cosine.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 4th 2023 05:54:45 am                                              #
# Modified   : Saturday February 4th 2023 11:33:15 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from scipy import sparse

from recsys.neighborhood.metric import CosineSimilarity


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.cosine
class TestCosineSimilarity:  # pragma: no cover
    # ============================================================================================ #
    @pytest.mark.skip("Currently working. Takes too long")
    def test_user_cosine(self, ratings, caplog):
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
        X = ratings.as_csr()
        cos = CosineSimilarity()
        c = cos(X, dimension="user")
        assert isinstance(c, sparse.csr_matrix)
        assert c.max() == 1
        assert c.min() == 0
        logger.debug(f"Shape of cosine similarity is {c.shape}.")
        logger.debug(f"There are  {c.nnz} nonzero elements.")

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
    # @pytest.mark.skip("Currently working. Takes too long")
    def test_item_cosine(self, ratings, caplog):
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
        X = ratings.as_csr()
        cos = CosineSimilarity()
        c = cos(X, dimension="item")
        assert isinstance(c, (sparse.csr_matrix, sparse.csc_matrix))
        assert c.max() == 1
        assert c.min() == 0
        logger.debug(f"Shape of cosine similarity is {c.shape}.")
        logger.debug(f"There are  {c.nnz} nonzero elements.")

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
