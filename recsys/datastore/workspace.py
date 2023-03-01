#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/datastore/workspace.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 09:09:12 pm                                               #
# Modified   : Tuesday February 28th 2023 11:41:14 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from atelier.utils.memory import get_size
from recsys.dataset.rating import RatingsDataset
from recsys.persistence.odb import ObjectDB
from recsys.datastore.asset import Repo, IDGen
from recsys.operator.data.sampling import UserRandomSampling
from recsys.persistence.io import IOService
from recsys.exceptions.workspace import WorkspaceException


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Workspace:
    name: str
    description: str
    location: str
    dataset: Repo = None
    experiment: Repo = None
    model: Repo = None
    size: int = None

    def size(self) -> None:
        self.size = get_size(self)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class WorkspaceConfig:  # pragma no cover
    name: str
    description: str
    dataset_name: str
    dataset_description: str
    dataset_filepath: str  # Filepath to the original dataset from which a sample will be taken.
    dataset_size: float  # The proportion of the original dataset to include


# ------------------------------------------------------------------------------------------------ #
class Builder(ABC):  # pragma no cover
    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        """Returns the workspace object"""

    @abstractmethod
    def reset(self) -> None:
        """Resets the builder object."""

    @abstractmethod
    def build(self, config: WorkspaceConfig) -> None:
        """Controls construction of the Workspace object."""

    @abstractmethod
    def build_workspace(self) -> None:
        """Initializes the workspace object."""

    @abstractmethod
    def _build_dataset_repo(self) -> None:
        """Constructs the Dataset repository and adds it to the workspace object"""

    @abstractmethod
    def _build_model_repo(self) -> None:
        """Constructs the Model repository and adds it to the workspace object"""

    @abstractmethod
    def _build_experiment_repo(self) -> None:
        """Constructs the Experiment repository and adds it to the workspace object"""


# ------------------------------------------------------------------------------------------------ #
class Workspacebuilder(Builder):
    def __init__(self, studio: str = "devstudio") -> None:
        super().__init__()
        self._studio = studio
        self._location = None
        self._workspace = None
        self._config = None
        self._idgen = None
        self._dataset_id = None
        self.reset()

    def workspace(self) -> Workspace:
        return self._workspace

    def reset(self) -> None:
        self._workspace = None
        self._location = None
        self._config = None

    def build_workspace(self, config: WorkspaceConfig) -> None:
        """Method that controls the construction process."""
        self._config = config
        self._set_location()
        self._workspace = Workspace(
            name=self._config.name, description=self._config.description, location=self._location
        )

    def _set_location(self) -> None:
        self._location = os.path.join(self._studio, self._config.name)

    def add_dataset_repo(self) -> None:
        """Creates the dataset repository."""
        filepath = os.path.join(self._location, "dataset.db")
        self._workspace.dataset = self._build_repo(filepath)

    def add_model_repo(self) -> None:
        """Creates the model repository."""
        filepath = os.path.join(self._location, "model.db")
        self._workspace.model = self._build_repo(filepath)

    def add_experiment_repo(self) -> None:
        """Creates the experiment repository."""
        filepath = os.path.join(self._location, "experiment.db")
        self._workspace.experiment = self._build_repo(filepath)

    def add_dataset(self) -> None:
        """Builds the dataset and adds it to the Dataset repo."""
        data = IOService.read(self._config.dataset)
        sampler = UserRandomSampling(frac=self._config.size)
        sample = sampler.run(data)
        dataset = RatingsDataset(
            name=self._config.dataset_name,
            description=self._config.dataset_description,
            data=sample,
        )
        self._dataset_id = self._workspace.dataset.add(asset=dataset)

    def check_workspace(self) -> bool:
        """Confirms the workspace has all components. Throws exception if missing component"""
        if self._workspace.dataset is None:
            msg = f"Workspace {self._config.name} missing dataset repository"
            self._log_raise(msg)
        elif self._workspace.experiment is None:
            msg = f"Workspace {self._config.name} missing model repository"
            self._log_raise(msg)
        elif self._workspace.experiment is None:
            msg = f"Workspace {self._config.name} missing experiment repository"
            self._log_raise(msg)
        elif self._workspace.dataset.exists(key=self._dataset_id) is False:
            msg = f"Workspace {self._config.name} dataset repository is missing the dataset."
            self._log_raise(msg)
        return True

    def _log_raise(self, msg: str) -> None:
        """Logs and raises WorkspaceException with the specified message."""
        self._logger.error(msg)
        raise WorkspaceException(msg)

    def _build_repo(self, filepath: str) -> Repo:
        db = ObjectDB(filepath=filepath)
        idgen = self._get_idgen()
        return Repo(database=db, idgen=idgen)

    def _get_idgen(self) -> None:
        if self._idgen is None:
            filepath = os.path.join(self._location, "idgen.db")
            db = ObjectDB(filepath=filepath)
            self._idgen = IDGen(database=db)
        return self._idgen


# ------------------------------------------------------------------------------------------------ #
class WorkspaceRepo(Repo):
    def __init__(self, database: ObjectDB) -> None:
        self._database = database

    def add(self, workspace: Workspace) -> None:
        """Adds an object to the repository"""
        self._database.insert(key=workspace.name, value=workspace)

    def get(self, key: str) -> Any:
        """Obtains an object from persistence by key."""
        return self._database.select(key=key)

    def update(self, workspace: Workspace) -> None:
        """Updates an object in storage"""
        self._database.update(key=workspace.name, value=workspace)

    def remove(self, key: str) -> None:
        """Removes an object from storage."""
        self._database.remove(key=key)

    def exists(self, key: str) -> bool:
        return self._database.exists(key)

    def info(self) -> None:
        """Prints the contents of storage."""
        inventory = []
        objects = self._database.selectall()
        for object in objects.values():
            d = {}
            d["class"] = object.__class.__.__name__
            d["name"] = object.name
            d["description"] = object.description
            d["location"] = object.location
            d["size"] = object.size
            inventory.append(d)

        inventory = pd.DataFrame(data=inventory, index=range(len(inventory)))
        print(inventory)
