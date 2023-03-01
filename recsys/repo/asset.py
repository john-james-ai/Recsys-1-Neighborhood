#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/repo/asset.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 10:44:08 pm                                              #
# Modified   : Wednesday March 1st 2023 03:34:09 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Asset Repo Module"""
from typing import Any

import pandas as pd

from recsys.assets.base import Asset
from recsys.repo.base import Repo
from recsys.persistence.odb import ObjectDB
from recsys.exceptions.repo import RepoException


# ------------------------------------------------------------------------------------------------ #
class AssetRepo(Repo):
    def __init__(self, database: ObjectDB, asset_type: type[Asset]) -> None:
        self._database = database
        self._asset_type = asset_type

    def add(self, asset: Asset) -> None:
        """Adds an object to the repository."""
        self._check_asset(asset)
        asset.save()
        self._database.insert(key=asset.name, value=asset)

    def get(self, name: str) -> Any:
        """Obtains an object from persistence by name."""
        return self._database.select(key=name)

    def update(self, asset: Asset) -> None:
        """Updates an object in storage"""
        self._check_asset(asset)
        asset.update()
        self._database.update(key=asset.name, value=asset)

    def remove(self, key: str) -> None:
        """Removes an object from storage."""
        self._database.remove(key=key)

    def reset(self) -> None:
        """Removes an object from storage."""
        self._database.clear()

    def exists(self, name: str) -> bool:
        return self._database.exists(key=name)

    def info(self) -> pd.DataFrame:
        """Prints the contents of storage."""
        inventory = []
        objects = self._database.selectall()
        for object in objects.values():
            d = {}
            d["id"] = object.id
            d["class"] = object.__class.__.__name__
            d["name"] = object.name
            d["description"] = object.description
            d["memory"] = object.memory
            d["created"] = object.created
            d["updated"] = object.updated
            inventory.append(d)

        inventory = pd.DataFrame(data=inventory, index=range(len(inventory)))
        return inventory

    def _check_asset(self, asset: Asset) -> None:
        """Ensures asset type is equal to the asset type for this repo."""
        if not isinstance(asset, self._asset_type):
            msg = f"Asset must be of type {self._asset_type}, not {type(asset)}."
            self._logger.error(msg)
            raise RepoException(msg)
