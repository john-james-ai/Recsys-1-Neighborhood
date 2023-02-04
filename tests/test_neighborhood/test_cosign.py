#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_neighborhood/test_cosign.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 2nd 2023 10:59:07 pm                                              #
# Modified   : Friday February 3rd 2023 10:57:41 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import numpy as np
from datetime import datetime
import pytest
import logging

from recsys.neighborhood.similarity import CosignSimilarity

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.sim
@pytest.mark.cos
class TestCosignSimilarity:  # pragma: no cover
    # ============================================================================================ #
    def test_user_similarity(self, ratings, caplog):
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
        sim = CosignSimilarity()
        sim.compute_user_similarity(ratings=ratings)
        user_sim = sim.user_similarity
        assert user_sim.name == "cosign_user_similarity_matrix"
        assert isinstance(user_sim.shape, tuple)
        assert isinstance(user_sim.size, int)
        assert isinstance(user_sim.memory, int)
        s = user_sim.get_similarity(a=5051, b=11916)
        assert np.isclose(s, 0.58)

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
    def test_item_similarity(self, ratings, caplog):
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
        sim = CosignSimilarity()
        sim.compute_item_similarity(ratings=ratings)
        item_sim = sim.item_similarity
        assert item_sim.name == "cosign_item_similarity_matrix"
        assert isinstance(item_sim.shape, tuple)
        assert isinstance(item_sim.size, int)
        assert isinstance(item_sim.memory, int)
        s = item_sim.get_similarity(a=5051, b=11916)
        assert np.isclose(
            s,
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
