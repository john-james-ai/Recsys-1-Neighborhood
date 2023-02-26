#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/repo.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 08:45:00 am                                               #
# Modified   : Sunday February 26th 2023 12:45:38 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Any

import pandas as pd

from recsys.persistence.odb import ObjectDB
from recsys import Asset


# ------------------------------------------------------------------------------------------------ #
class Repo:
    def __init__(self, database: ObjectDB) -> None:
        self._database = database

    def add(self, asset: Asset) -> None:
        """Adds an object to the repository"""
        self._database.insert(key=asset.name, value=asset)

    def get(self, key: str) -> Any:
        """Obtains an object from persistence by key."""
        return self._database.select(key=key)

    def update(self, asset: Asset) -> None:
        """Updates an object in storage"""
        self._database.update(key=asset.name, value=asset)

    def remove(self, key: str) -> None:
        """Removes an object from storage."""
        self._database.remove(key=key)

    def print(self) -> pd.DataFrame:
        """Prints the contents of storage."""
        pass
