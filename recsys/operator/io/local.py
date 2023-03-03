#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/io/local.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 05:57:39 am                                             #
# Modified   : Thursday March 2nd 2023 08:31:54 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Local IO Module"""
from typing import Any

from recsys import Operator
from atelier.persistence.io import IOService


# ------------------------------------------------------------------------------------------------ #
#                                      FILE READER                                                 #
# ------------------------------------------------------------------------------------------------ #
class FileReader(Operator):
    """Reads data from a file.

    Currently supports csv, yaml, excel, and pickle file formats. The format is inferred from
    the filepath extension.

    Args:
        filepath (str): Path of file to read.
    """

    def __init__(self, filepath: str, fio: IOService = IOService) -> None:
        self._filepath = filepath
        self._fio = fio

    def run(self, *args, **kwargs) -> Any:
        return self._fio.read(self._filepath)


# ------------------------------------------------------------------------------------------------ #
#                                      FILE WRITER                                                 #
# ------------------------------------------------------------------------------------------------ #
class FileWriter(Operator):
    """Writes data to a file.

    Currently supports csv, yaml, excel, and pickle file formats. The format is inferred from
    the filepath extension. The data to write is passed into the run method at runtime.

    Args:
        filepath (str): Path of file to read.
    """

    def __init__(self, filepath: str, fio: IOService = IOService) -> None:

        self._filepath = filepath
        self._fio = fio

    def run(self, data: Any) -> None:
        self._fio.write(filepath=self._filepath, data=data)


# ------------------------------------------------------------------------------------------------ #
#                                     CONVERT FILE                                                 #
# ------------------------------------------------------------------------------------------------ #
class ConvertFile(Operator):
    """Converts a file between the supported formats.

    The source and destination file formats are inferred from the filepath extensions.

    Args:
        source (str): Source filepath
        destination (str): Destination filepath
        force (bool): Whether to force execution.
    """

    def __init__(
        self, source: str, destination: str, fio: IOService = IOService, force: bool = False
    ) -> None:
        self._source = source
        self._destination = destination
        self._fio = fio
        self._force = force

    def run(self, *args, **kwargs) -> None:
        """Converts the file"""
        data = self._fio.read(filepath=self._source)
        self._fio.write(filepath=self._destination, data=data)
