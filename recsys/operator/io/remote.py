#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/remote.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:35:10 pm                                            #
# Modified   : Saturday February 25th 2023 08:40:01 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
import os
import requests
from tqdm import tqdm

from recsys import operator


# ------------------------------------------------------------------------------------------------ #
#                                   ZIP DOWNLOADER                                                 #
# ------------------------------------------------------------------------------------------------ #
class ZipDownloader(operator.FileOperator):
    """Downloads a zip file from a website.

    Args:
        source (str): The URL to the zip file resource
        destination (str): A filename into which the zip file will be stored.
        force (bool): Whether to force execution.
    """

    def __init__(
        self, source: str, destination: str, chunk_size: int = 1024, force: bool = False
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._chunk_size = chunk_size

    def run(self, *args, **kwargs) -> None:
        """Downloads a zipfile."""

        resp = requests.get(self._source, stream=True)
        total = int(resp.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(self._destination), exist_ok=True)
        with open(self._destination, "wb") as file, tqdm(
            desc=self._destination,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=self._chunk_size):
                size = file.write(data)
                bar.update(size)
