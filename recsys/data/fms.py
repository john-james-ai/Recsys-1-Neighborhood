#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/fms.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 19th 2023 03:47:35 am                                               #
# Modified   : Monday February 20th 2023 11:25:03 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from typing import Any
from dependency_injector.wiring import Provide, inject

from recsys.data.base import FMSBase
from recsys.data.fio import IOService
from recsys.container import Recsys


# ------------------------------------------------------------------------------------------------ #
class FMS(FMSBase):
    """File Management Services.

    Args:
        config (dict): File service configuration
        io (IOService): IO parser
    """

    __format = "pkl"

    @inject
    def __init__(self, io: IOService = Provide[Recsys.data.io]) -> None:
        super().__init__()
        self._io = io

    def read(self, filepath: str) -> Any:
        """Read the file"""
        return self._io.read(filepath)

    def write(self, data: Any, filepath: str) -> Any:
        """Write data to file"""
        self._io.write(data=data, filepath=filepath)

    def get_filepath(self, name: str, stage: str) -> str:
        """Returns the filepath for the provided name, and stage."""
        # The current environment is stored in an environment variable.
        return os.path.join("data", "movielens25m", stage) + "_" + name + ".pkl"
