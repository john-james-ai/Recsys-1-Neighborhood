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
# Modified   : Sunday February 19th 2023 04:37:17 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv
from dependency_injector.wiring import Provide, inject

from recsys.data.base import FMS
from recsys.data.fio import IOService
from recsys.container import Recsys


# ------------------------------------------------------------------------------------------------ #
class FileService(FMS):
    """File Service class providing read write behavior.

    Args:
        config (dict): File service configuration
        io (IOService): IO parser
    """

    __format = "pkl"

    @inject
    def __init__(self, config: dict, io: IOService = Provide[Recsys.data.io]) -> None:
        super().__init__(config=config, io=io)

    def get_filepath(self, name: str, stage: str) -> str:
        """Returns the filepath for the provided name, stage and current environment."""
        # The current environment is stored in an environment variable.
        load_dotenv()
        env = os.environ.get("ENV")
        basedir = self._config.get(env)
        name = name + "." + self._config.get("format", FileService.__format)
        return os.path.join(basedir, stage, name)


# ------------------------------------------------------------------------------------------------ #
class ModelService(FMS):
    """Model Service class providing read write behavior.

    Args:
        config (dict): File service configuration
        io (IOService): IO parser
    """

    __format = "pkl"

    @inject
    def __init__(self, config: dict, io: IOService = Provide[Recsys.data.io]) -> None:
        super().__init__(config=config, io=io)

    def get_filepath(self, name: str, stage: str) -> str:
        """Returns the filepath for the provided name, stage and current environment."""
        # The current environment is stored in an environment variable.
        load_dotenv()
        env = os.environ.get("ENV")
        basedir = self._config.get(env)
        name = name + "." + self._config.get("format", ModelService.__format)
        return os.path.join(basedir, stage, name)
