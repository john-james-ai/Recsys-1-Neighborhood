#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataset/movielens.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 03:29:59 pm                                                  #
# Modified   : Friday March 17th 2023 08:03:32 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dataclasses import dataclass

from recsys.dataset.base import Dataset
from recsys.operator.io.remote import ZipDownloader
from recsys.operator.io.compress import ZipExtractor
from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
@dataclass
class MovieLens(Dataset):
    source: str = None
    destination: str = None
    filename: str = None
    directory: str = None
    force: bool = False

    def fetch_data(self) -> None:
        """Downloads a data source and extracts the interaction data."""
        downloader = ZipDownloader(source=self.source, destination=self.destination)
        downloader.execute()

        extractor = ZipExtractor(
            source=self.destination, destination=self.directory, member=self.filename
        )
        extractor.execute()


@dataclass
class MovieLens1M(MovieLens):
    """Encapsulates data and operations for the MovieLens25M data source"""

    source: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    destination: str = "data/ext/ml-1m.zip"
    filename: str = "ratings.dat"
    directory: str = "data/movielens1m/raw"
    force: bool = False

    def fetch_data(self) -> None:
        super().fetch_data()
        ratings = IOService.read(os.path.join(self.directory, self.filename), sep="::")
        IOService.write(filepath=os.path.join(self.directory, "ratings.pkl"), data=ratings)
        return ratings


@dataclass
class MovieLens10M(MovieLens):
    """Encapsulates data and operations for the MovieLens25M data source"""

    source: str = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    destination: str = "data/ext/ml-10m.zip"
    filename: str = "ratings.dat"
    directory: str = "data/movielens10m/raw"
    force: bool = False

    def fetch_data(self) -> None:
        super().fetch_data()
        ratings = IOService.read(os.path.join(self.directory, self.filename), sep="::")
        IOService.write(filepath=os.path.join(self.directory, "ratings.pkl"), data=ratings)
        return ratings


@dataclass
class MovieLens25M(MovieLens):
    """Encapsulates data and operations for the MovieLens25M data source"""

    source: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    destination: str = "data/ext/ml-25m.zip"
    filename: str = "ratings.csv"
    directory: str = "data/movielens25m/raw"
    force: bool = False

    def fetch_data(self) -> None:
        super().fetch_data()
        ratings = IOService.read(os.path.join(self.directory, self.filename))
        IOService.write(filepath=os.path.join(self.directory, "ratings.pkl"), data=ratings)
        return ratings
