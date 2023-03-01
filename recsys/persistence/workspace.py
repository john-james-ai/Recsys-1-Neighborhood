#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/workspace.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 09:09:12 pm                                               #
# Modified   : Tuesday February 28th 2023 09:03:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from recsys.persistence.cache import Cache
from recsys.persistence.odb import ObjectDB, CacheDB, IDGen
from recsys.persistence.repo import Repo, IDGen


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Workspace:
    name: str = None
    description: str = None
    dataset: Repo = None
    experiment: Repo = None
    model: Repo = None
    cache: Cache = None


# ------------------------------------------------------------------------------------------------ #
class WorkspaceConfig:
    name: str
    description: str
    ttl: int


# ------------------------------------------------------------------------------------------------ #
class Builder:
    @property
    @abstractmethod
    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def workspace(self) -> Workspace:
        """Returns the workspace object"""

    @abstractmethod
    def build_dataset_repo(self) -> None:
        """Constructs the Dataset repostitory"""

    @abstractmethod
    def build_model_repo(self) -> None:
        """Constructs the model repostitory"""

    @abstractmethod
    def build_experiment_repo(self) -> None:
        """Constructs the experiment repostitory"""

    @abstractmethod
    def build_cache(self) -> None:
        """Builds the cache for the workspace"""


# ------------------------------------------------------------------------------------------------ #
class Workspacebuilder(Builder):
    def __init__(self, location: str = "devstudio") -> None:
        super().__init__()
        self._config = None
        self._location = location
        self.reset()

    def workspace(self) -> Workspace:
        return self._workspace

    def reset(self) -> None:
        self._workspace = Workspace()
        self._workspace_location = None
        self._idgen = None

    def build(self, config: WorkspaceConfig) -> None:
        self._set_config(config)
        self._build_dataset_repo()
        self._build_experiment_repo()
        self._build_model_repo()
        self._build_cache()

    def _build_dataset_repo(self) -> None:
        filepath = os.path.join(self._location, "dataset.db")
        self._workspace.dataset = self._build_repo(filepath)

    def _build_model_repo(self) -> None:
        filepath = os.path.join(self._location, "model.db")
        self._workspace.model = self._build_repo(filepath)

    def _build_experiment_repo(self) -> None:
        filepath = os.path.join(self._location, "experiment.db")
        self._workspace.experiment = self._build_repo(filepath)

    def _build_cache(self) -> None:
        filepath = os.path.join(self._location, "cache.db")

    def _set_config(self, config: WorkspaceConfig) -> None:
        if isinstance(config, WorkspaceConfig):
            self._config = config
            self._workspace_location = os.path.join(self._location, config.name)
        else:
            msg = f"The config object must be a WorkspaceConfig type, not {type(config)}."
            self._logger.error(msg)

    def _build_repo(self, filepath: str) -> Repo:
        db = ObjectDB(filepath=filepath)
        idgen = self._get_idgen()
        return Repo(database=db, idgen=idgen)

    def _get_idgen(self) -> None:
        filepath = os.path.join(self._location, "idgen.db")
        db = ObjectDB(filepath=filepath)
        self._idgen = IDGen(database=db)
