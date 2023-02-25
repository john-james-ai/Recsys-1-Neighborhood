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
# Modified   : Saturday February 25th 2023 08:26:12 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Local IO Module"""
from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                                      FILE READER                                                 #
# ------------------------------------------------------------------------------------------------ #
class FileReader(Operator):
    """Reads data from a file.

    Args:

    """


# ------------------------------------------------------------------------------------------------ #
#                                      COPY FILE                                                   #
# ------------------------------------------------------------------------------------------------ #
class CopyFile(Operator):
    """Copies a file.

    Args:
        source (str): Source filepath
        destination (str): Destination filepath
        force (bool): Whether to force execution.
    """

    def __init__(self, source: str, destination: str, force: bool = False) -> None:
        super().__init__(source=source, destination=destination, force=force)

    def run(self, *args, **kwargs) -> None:
        """Extracts the contents"""
        data = self._fio.read(filepath=self._source)
        self._fio.write(filepath=self._destination, data=data)
