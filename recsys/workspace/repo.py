#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workspace/repo.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 10:44:08 pm                                              #
# Modified   : Wednesday March 1st 2023 07:04:55 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Workspace Repo Module"""
import pandas as pd

from recsys.workspace.workspace import Workspace
from recsys.repo.base import Repo
from recsys.persistence.odb import ObjectDB


# ------------------------------------------------------------------------------------------------ #
class WorkspaceRepo(Repo):
    def __init__(self, database: ObjectDB) -> None:
        self._database = database

    def add(self, workspace: Workspace) -> None:
        """Adds an object to the repository."""
        self._database.insert(key=workspace.name, value=workspace)

    def get(self, name: str) -> Workspace:
        """Obtains an object from persistence by name."""
        return self._database.select(key=name)

    def update(self, workspace: Workspace) -> None:
        """Updates an object in storage"""
        self._database.update(key=workspace.name, value=workspace)

    def remove(self, name: str) -> None:
        """Removes an object from storage."""
        self._database.remove(key=name)

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
