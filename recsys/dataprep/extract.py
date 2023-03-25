#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/extract.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 05:55:47 am                                             #
# Modified   : Monday March 20th 2023 09:48:55 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Compression Module"""
import os
from zipfile import ZipFile
import logging

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                                   ZIP EXTRACTOR                                                  #
# ------------------------------------------------------------------------------------------------ #
class ZipExtractOperator(Operator):
    """Extracts Zipfile contents.

    Args:
        source (str): Path to the zipfile
        destination (str): The extract directory
        member (str): The member to extract. If None, all members will be extracted.
        force (bool): Whether to force execution.
    """

    __name = "zip_extract_operator"
    __desc = "Extracts files from zip archives."

    def __init__(
        self, source: str, destination: str, member: str = None, force: bool = False
    ) -> None:
        super().__init__()
        self._source = source
        self._destination = destination
        self._force = force
        self._member = member
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __call__(self, *args, **kwargs) -> None:
        """Extracts the contents"""

        if not self._skip(endpoint=self._destination):

            os.makedirs(self._destination, exist_ok=True)

            with ZipFile(self._source, mode="r") as zip:
                for zip_info in zip.infolist():
                    if zip_info.filename[-1] == "/":
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    if self._member is not None:
                        if zip_info.filename == self._member:
                            zip.extract(zip_info, self._destination)
                        else:
                            continue
                    else:
                        zip.extract(zip_info, self._destination)
            self._logger.debug(f"Extracted zip archive from {self._source} to {self._destination}")
