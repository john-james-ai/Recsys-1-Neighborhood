#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/local.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 05:57:39 am                                             #
# Modified   : Saturday February 25th 2023 08:38:58 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Local IO Module"""
from typing import Any
from dependency_injector.wiring import Provide, inject

from recsys import operator
from recsys.container import Recsys
from recsys.io.service import IOService


# ------------------------------------------------------------------------------------------------ #
#                                      FILE READER                                                 #
# ------------------------------------------------------------------------------------------------ #
class FileReader(operator.FileOperator):
    """Reads data from a file.

    Currently supports csv, yaml, excel, and pickle file formats. The format is inferred from
    the filepath extension.

    Args:
        filepath (str): Path of file to read.
    """

    @inject
    def __init__(self, filepath: str, fio: IOService = Provide[Recsys.services.fio]) -> None:
        self._filepath = filepath
        self._fio = fio

    def run(self, *args, **kwargs) -> Any:
        return self._fio.read(self._filepath)


# ------------------------------------------------------------------------------------------------ #
#                                      FILE WRITER                                                 #
# ------------------------------------------------------------------------------------------------ #
class FileWriter(operator.FileOperator):
    """Writes data to a file.

    Currently supports csv, yaml, excel, and pickle file formats. The format is inferred from
    the filepath extension. The data to write is passed into the run method at runtime.

    Args:
        filepath (str): Path of file to read.
    """

    @inject
    def __init__(self, filepath: str, fio: IOService = Provide[Recsys.services.fio]) -> None:

        self._filepath = filepath
        self._fio = fio

    def run(self, data: Any) -> None:
        self._fio.write(filepath=self._filepath, data=data)


# ------------------------------------------------------------------------------------------------ #
#                                     CONVERT FILE                                                 #
# ------------------------------------------------------------------------------------------------ #
class ConvertFile(operator.FileOperator):
    """Converts a file between the supported formats.

    The source and destination file formats are inferred from the filepath extensions.

    Args:
        source (str): Source filepath
        destination (str): Destination filepath
        force (bool): Whether to force execution.
    """

    def __init__(self, source: str, destination: str, force: bool = False) -> None:
        super().__init__(source=source, destination=destination, force=force)

    def run(self, *args, **kwargs) -> None:
        """Converts the file"""
        data = self._fio.read(filepath=self._source)
        self._fio.write(filepath=self._destination, data=data)
