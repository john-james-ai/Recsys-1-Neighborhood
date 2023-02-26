#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/compress.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:35:10 pm                                            #
# Modified   : Saturday February 25th 2023 08:39:44 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
from zipfile import ZipFile

from recsys import operator


# ------------------------------------------------------------------------------------------------ #
#                                   ZIP EXTRACTOR                                                  #
# ------------------------------------------------------------------------------------------------ #
class ZipExtractor(operator.FileOperator):
    """Extracts Zipfile contents.

    Args:
        source(str): Path to the zipfile
        destination (str): The extract directory
        force (bool): Whether to force execution.
    """

    def __init__(self, source: str, destination: str, force: bool = False) -> None:
        super().__init__(source=source, destination=destination, force=force)

    def run(self, *args, **kwargs) -> None:
        """Extracts the contents"""

        with ZipFile(self._source, mode="r") as archive:
            archive.extractall(self._destination)
