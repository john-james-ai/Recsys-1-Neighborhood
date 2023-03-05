#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/io/compress.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:35:10 pm                                            #
# Modified   : Sunday March 5th 2023 01:30:42 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
import os
from zipfile import ZipFile

from recsys.operator.base import Operator, Artifact


# ------------------------------------------------------------------------------------------------ #
#                                   ZIP EXTRACTOR                                                  #
# ------------------------------------------------------------------------------------------------ #
class ZipExtractor(Operator):
    """Extracts Zipfile contents.

    Args:
        source(str): Path to the zipfile
        destination (str): The extract directory
        member (str): The member to extract. If None, all members will be extracted.
        force (bool): Whether to force execution.
    """

    def __init__(
        self, source: str, destination: str, member: str = None, force: bool = False
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._member = member
        filepath = os.path.join(self._destination, member)
        self._artifact = Artifact(isfile=True, path=filepath, uripath="data")

    def execute(self, *args, **kwargs) -> None:
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
