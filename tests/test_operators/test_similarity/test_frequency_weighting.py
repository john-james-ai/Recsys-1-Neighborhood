#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /tests/test_operators/test_similarity/test_frequency_weighting.py                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 08:09:28 pm                                                 #
# Modified   : Sunday March 12th 2023 03:34:21 am                                                  #
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

from recsys.data.array import Array
from recsys.operator.matrix.weights import FrequencyWeightedMatrixFactory
from recsys.operator.matrix.interaction import InteractionMatrixFactory
from recsys.services.io import IOService

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

INTERACTION = "tests/testdata/operators/data_operators/factories/interaction.pkl"

USER_WEIGHTS = "tests/testdata/operators/similarity/factories/weights/frequency/user_weights.pkl"
ITEM_WEIGHTS = "tests/testdata/operators/similarity/factories/weights/frequency/item_weights.pkl"


@pytest.mark.freqweights
class TestWeighting:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, dataset, caplog):
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
        shutil.rmtree(os.path.dirname(USER_WEIGHTS), ignore_errors=True)
        factory = InteractionMatrixFactory(
            name="test_interaction_matrix",
            description="Test Interaction Matrix",
            destination=INTERACTION,
        )
        factory.execute(data=dataset)

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
    def test_user_weights(self, dataset, interaction, caplog):
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
        factory = FrequencyWeightedMatrixFactory(
            name="user_weights",
            description="User Frequency Weighting Matrix",
            source=INTERACTION,
            destination=USER_WEIGHTS,
            dim="user",
        )
        weights = factory.execute(data=interaction)
        assert isinstance(weights, Array)
        assert weights.max <= np.log(dataset.n_users / 1)
        assert weights.min >= np.log(dataset.n_users / 1001)

        with pytest.raises(ValueError):
            factory = FrequencyWeightedMatrixFactory(
                name="user_weights",
                description="User Significance Weighting Matrix",
                source=INTERACTION,
                destination=USER_WEIGHTS,
                dim="NOTVALID",
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
    def test_user_weight_sol(self, caplog):
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
        weights = IOService.read(USER_WEIGHTS)
        assert weights.shape == (47171, 1)

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
    def test_item_weights(self, dataset, interaction, caplog):
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
        factory = FrequencyWeightedMatrixFactory(
            name="item_weights",
            description="Item Frequency Weighting Matrix",
            source=INTERACTION,
            destination=ITEM_WEIGHTS,
            dim="item",
        )
        weights = factory.execute(data=interaction)
        assert isinstance(weights, Array)
        assert weights.max <= np.log(dataset.n_items / 1)
        assert weights.min >= np.log(dataset.n_items / 1000)

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
    def test_item_weight_sol(self, caplog):
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
        weights = IOService.read(ITEM_WEIGHTS)
        assert weights.shape == (141779, 1)
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
