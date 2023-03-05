#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/etl/extract.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 07:40:00 am                                                 #
# Modified   : Saturday March 4th 2023 10:21:44 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Extracts data from Grouplens site"""
import os
import mlflow
import click
import logging

from recsys.operator.io.remote import ZipDownloader
from recsys.operator.io.compress import ZipExtractor

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@click.command(
    help="""Given a URL, downloads the data and extracts the data from the zipfile
    Artifacts will be registered in MLflow , and registers the artifacts in MLFlow"""
)
@click.option(
    "--source",
    default="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    help="The URL from which the data will be downloaded.",
)
@click.option("--destination", default="data/ext", help="The download directory")
@click.option(
    "--raw_dir", default="data/raw", help="The directory into which the raw data will be extracted."
)
@click.option("--member", default="ratings.csv", help="The ratings file to be extracted.")
def extract(source, destination, raw_dir, member):
    with mlflow.start_run() as mlrun:  # noqa F841
        # Download data
        downloader = ZipDownloader(source=source, destination=destination)
        downloader.execute()
        # Extract member from zipfile
        extractor = ZipExtractor(source=destination, destination=raw_dir, member=member)
        extractor.execute()
        # Log artifact
        ratings_filepath = os.path.join(raw_dir, member)
        logger.info(f"Uploading raw ratings: {ratings_filepath}.")
        mlflow.log_artifacts(ratings_filepath, "ratings-raw_filepath")


if __name__ == "__main__":
    extract()
