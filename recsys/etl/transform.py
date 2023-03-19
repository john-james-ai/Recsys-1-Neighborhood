#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/etl/transform.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 07:40:00 am                                                 #
# Modified   : Saturday March 18th 2023 07:43:23 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Transforms MovieLens ratings dataset to pickle format."""
import os
import mlflow
import click
import logging

from recsys import IOService

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@click.command(
    help="""Converts ratings dataset from csv to pickle format and registers the artifacts in MLFlow"""
)
@click.option("--source", default="data/raw/ratings.csv", help="The path to the ratings.csv file.")
@click.option("--destination", default="data/raw/ratings.pkl", help="Filepath for pickle file.")
def transform(source, destination):
    with mlflow.start_run() as mlrun:  # noqa F841

        # Perform the conversion
        ratings = IOService.read(filepath=source)
        IOService.write(filepath=destination, data=ratings)
        source_filename = os.path.basename(source)
        destination_filename = os.path.basename(destination)
        logger.info(f"Converted {source_filename} to {destination_filename}")

        # Log artifact
        logger.info(f"Uploading transformed ratings file: {destination}.")
        mlflow.log_artifacts(destination, "ratings-transformed_filepath")


if __name__ == "__main__":
    transform()
