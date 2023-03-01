#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/datastore/asset.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 10:44:08 pm                                              #
# Modified   : Tuesday February 28th 2023 11:25:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
import os
import logging
from dotenv import load_dotenv
from typing import Any

import pandas as pd
from recsys.assets.base import Asset
from recsys.datastore.repo import Repo
from recsys.persistence.odb import ObjectDB


# ------------------------------------------------------------------------------------------------ #
class AssetRepo(Repo):
    def __init__(self, database: ObjectDB, idgen: IDGen) -> None:
        self._database = database
        self._idgen = idgen

    def add(self, asset: Asset) -> str:
        """Adds an object to the repository and returns the id."""
        asset.id = self._idgen.getid(asset)
        asset.save()
        self._database.insert(key=asset.oid, value=asset)
        return asset.id

    def get(self, key: str) -> Any:
        """Obtains an object from persistence by key."""
        return self._database.select(key=key)

    def update(self, asset: Asset) -> None:
        """Updates an object in storage"""
        asset.update()
        self._database.update(key=asset.oid, value=asset)

    def remove(self, key: str) -> None:
        """Removes an object from storage."""
        self._database.remove(key=key)

    def exists(self, key: str) -> bool:
        return self._database.exists(key)

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
            d["create"] = object.created
            d["updated"] = object.updated
            inventory.append(d)

        inventory = pd.DataFrame(data=inventory, index=range(len(inventory)))
        return inventory


# ------------------------------------------------------------------------------------------------ #
class IDGen:
    def __init__(self, database: ObjectDB) -> None:
        self._database = database
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def getid(self, asset: Asset) -> int:
        key = self._get_key(asset)
        try:
            id = self._database.select(key=key)
            self._database.update(key=key, value=id + 1)
        except ObjectDB.Database.ObjectNotFoundError:
            id = 1
            self._database.insert(key=key, value=2)
        return id

    def reset(self, asset: Asset) -> None:
        key = self._get_key(asset)
        try:
            self._database.delete(key)
        except ObjectDB.Database.ObjectNotFoundError:
            pass
        self._database.insert(key=key, value=1)

    def _get_key(self, asset: Asset) -> str:
        load_dotenv()
        workspace = os.getenv(key="WORKSPACE")
        try:
            return workspace + "_" + asset.__class__.__name__.lower()
        except Exception as e:
            self._logger.error(e)
            raise
