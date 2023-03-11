#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /tests/test_operators/test_data_operators/test_factories.py                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 9th 2023 06:16:38 pm                                                 #
# Modified   : Thursday March 9th 2023 08:20:35 pm                                                 #
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

from recsys.services.io import IOService
from recsys.data.dataset import Dataset
from recsys.data.matrix import Matrix
from recsys.operator.dataset.factory import DatasetFactory
from recsys.operator.matrix.interaction import InteractionMatrixFactory
from recsys.operator.matrix.cooccurrence import CooccurrenceMatrixFactory


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

INPUT = "tests/testdata/operators/data_operators/sampling/temporaralthreshold/ratings_random_temporal_sampling_1000.pkl"
OUTPUT_OBJECTS = "tests/testdata/operators/data_operators/factories"
DATASET_OUTPUT = os.path.join(OUTPUT_OBJECTS, "dataset.pkl")
INTERACTION_OUTPUT = os.path.join(OUTPUT_OBJECTS, "interaction.pkl")
COOCCURRENCE_OUTPUT_USER = os.path.join(OUTPUT_OBJECTS, "user_interaction.pkl")
COOCCURRENCE_OUTPUT_ITEM = os.path.join(OUTPUT_OBJECTS, "item_interaction.pkl")


@pytest.mark.factory
class TestFactory:  # pragma: no cover
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
        shutil.rmtree(OUTPUT_OBJECTS, ignore_errors=True)

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
    def test_dataset(self, caplog):
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
        factory = DatasetFactory(
            name="test_dataset",
            description="Test Sampled Dataset",
            source=INPUT,
            destination=DATASET_OUTPUT,
        )
        factory.execute()
        ds = IOService.read(DATASET_OUTPUT)
        assert os.path.exists(DATASET_OUTPUT)
        assert isinstance(ds, Dataset)

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
    def test_interaction(self, dataset, caplog):
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
        factory = InteractionMatrixFactory(
            name="test_interaction_matrix",
            description="Test Interaction Matrix",
            destination=INTERACTION_OUTPUT,
        )
        interaction = factory.execute(data=dataset)
        assert os.path.exists(INTERACTION_OUTPUT)
        assert isinstance(interaction, Matrix)

        interaction = factory.execute(data=dataset)  # Should be skipped.

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
    def test_user_cooccurrence(self, interaction, caplog):
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
        factory = CooccurrenceMatrixFactory(
            name="test_user_cooccurrence_matrix",
            description="Test User Cooccurrence Matrix",
            axis=0,
            destination=COOCCURRENCE_OUTPUT_USER,
        )
        cooccurrence = factory.execute(data=interaction)
        assert os.path.exists(COOCCURRENCE_OUTPUT_USER)
        assert isinstance(cooccurrence, Matrix)

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
    def test_item_cooccurrence(self, interaction, caplog):
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
        factory = CooccurrenceMatrixFactory(
            name="test_item_cooccurrence_matrix",
            description="Test Item Cooccurrence Matrix",
            axis=1,
            destination=COOCCURRENCE_OUTPUT_ITEM,
        )
        cooccurrence = factory.execute(data=interaction)
        assert os.path.exists(COOCCURRENCE_OUTPUT_ITEM)
        assert isinstance(cooccurrence, Matrix)

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
