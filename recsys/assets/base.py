#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/assets/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 01:00:36 pm                                              #
# Modified   : Wednesday March 1st 2023 03:18:30 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
import os
from abc import ABC
import logging
from dotenv import load_dotenv
from datetime import datetime

from atelier.utils.memory import get_size


# ------------------------------------------------------------------------------------------------ #
class Asset(ABC):  # pragma: no cover
    def __init__(self, name: str, description: str) -> None:

        load_dotenv()

        self._name = name
        self._description = description

        self._created = datetime.now()
        self._saved = None
        self._updated = None
        self._memory = None

        self._workspace = os.getenv("WORKSPACE")

        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def memory(self) -> int:
        if self._memory is None:
            self._memory = get_size(self)
        return self._memory

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def saved(self) -> datetime:
        return self._saved

    @property
    def updated(self) -> datetime:
        return self._updated

    def save(self) -> None:
        self._memory = get_size(self)
        self._saved = datetime.now()

    def update(self) -> None:
        self._memory = get_size(self)
        self._updated = datetime.now()
