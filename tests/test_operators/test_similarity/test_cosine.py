#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /tests/test_operators/test_similarity/test_cosine.py                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 08:09:28 pm                                                 #
# Modified   : Saturday March 11th 2023 02:11:28 pm                                                #
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

from scipy.sparse import csr_matrix, csc_matrix

from recsys.data.matrix import Matrix
from recsys.operator.matrix.similarity import SimilarityMatrixFactory


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


USER_COSINE_SIMILARITY = "tests/testdata/operators/similarity/factories/user_cosine.pkl"
ITEM_COSINE_SIMILARITY = "tests/testdata/operators/similarity/factories/item_cosine.pkl"

USER_PEARSON_SIMILARITY = "tests/testdata/operators/similarity/factories/user_pearson.pkl"
ITEM_PEARSON_SIMILARITY = "tests/testdata/operators/similarity/factories/item_pearson.pkl"


@pytest.mark.cosine
class TestCosine:  # pragma: no cover
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
        shutil.rmtree(os.path.dirname(USER_COSINE_SIMILARITY), ignore_errors=True)
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
    # @pytest.mark.skip()
    def test_user_cosine(self, dataset, caplog):
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
        factory = SimilarityMatrixFactory(
            name="cosine_similarity",
            description="Cosine Similarity",
            dim="user",
            metric="cosine",
            destination=USER_COSINE_SIMILARITY,
        )
        cosine = factory.execute(data=dataset)
        csr = cosine.to_csr()
        assert isinstance(cosine, Matrix)
        assert isinstance(csr, csr_matrix)
        assert csr.max() <= 1.01
        assert csr.min() >= -1.01

        with pytest.raises(ValueError):
            factory = SimilarityMatrixFactory(
                name="cosine_similarity",
                description="Cosine Similarity",
                dim="df",
                metric="cosine",
                destination=USER_COSINE_SIMILARITY,
            )

        with pytest.raises(ValueError):
            factory = SimilarityMatrixFactory(
                name="cosine_similarity",
                description="Cosine Similarity",
                dim="user",
                metric="notvalid",
                destination=USER_COSINE_SIMILARITY,
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

    # ============================================================================================ #
    # @pytest.mark.skip()
    def test_item_cosine(self, dataset, caplog):
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
        factory = SimilarityMatrixFactory(
            name="cosine_similarity",
            description="Cosine Similarity",
            dim="item",
            metric="cosine",
            destination=ITEM_COSINE_SIMILARITY,
        )
        cosine = factory.execute(data=dataset)
        csc = cosine.to_csc()
        assert isinstance(cosine, Matrix)
        assert isinstance(csc, csc_matrix)
        assert csc.max() <= 1.01
        assert csc.min() >= -1.01

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
