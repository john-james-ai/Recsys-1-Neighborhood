#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/io/compress.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:35:10 pm                                            #
# Modified   : Friday March 3rd 2023 01:46:04 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
from zipfile import ZipFile

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                                   ZIP EXTRACTOR                                                  #
# ------------------------------------------------------------------------------------------------ #
class ZipExtractor(Operator):
    """Extracts Zipfile contents.

    Args:
        source(str): Path to the zipfile
        destination (str): The extract directory
        force (bool): Whether to force execution.
    """

    def __init__(self, source: str, destination: str, force: bool = False) -> None:
        super().__init__(source=source, destination=destination, force=force)

    def execute(self, *args, **kwargs) -> None:
        """Extracts the contents"""

        if not self._skip(endpoint=self._destination):
            with ZipFile(self._source, mode="r") as archive:
                archive.extractall(self._destination)
