#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/etl/load.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 07:40:00 am                                                 #
# Modified   : Thursday March 9th 2023 07:19:50 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Loads training and test sets for production and development modes."""
import os
import mlflow
import click
import logging

from recsys.services.io import IOService
from recsys.operator.dataset.split import TemporalTrainTestSplit
from recsys.operator.dataset.sampling import UserStratifiedRandomSampling

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@click.command(help="""Loads training and test sets.""")
@click.option("--source", help="The path to the input dataset.")
@click.option(
    "--destination", help="Directory into which the training and test sets will be saved."
)
@click.option("--sample_size", help="The proportion of the source to sample.")
@click.option(
    "--train_size", default=0.8, help="Proportion of sample dataset to allocate to training set."
)
@click.option("--train_filename", default="train.pkl", help="Filename for training sets.")
@click.option("--test_filename", default="test.pkl", help="Filename for test sets.")
def load(source, destination, sample_size, train_size, train_filename, test_filename):
    with mlflow.start_run() as mlrun:  # noqa F841

        # Sample or retrieve the data
        if sample_size == 1:
            data = IOService.read(source)
        else:
            sampler = UserStratifiedRandomSampling(frac=sample_size, source=source)
            data = sampler.execute()

        # Create Production Training and Test Sets
        splitter = TemporalTrainTestSplit(
            destination=destination,
            train_size=train_size,
            train_filename=train_filename,
            test_filename=test_filename,
        )
        splitter.execute(data=data)

        # Log development training and test sets
        train_filepath = os.path.join(destination, train_filename)
        test_filepath = os.path.join(destination, test_filename)
        logger.info(f"Uploading development training set: {train_filepath} to MLFlow artifacts.")
        mlflow.log_artifacts(train_filepath, "datasets")
        logger.info(f"Uploading development test set: {test_filepath} to MLFlow artifacts..")
        mlflow.log_artifacts(test_filepath, "datasets")


if __name__ == "__main__":
    load()
