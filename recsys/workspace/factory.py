#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workspace/factory.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 06:35:52 am                                                #
# Modified   : Wednesday March 1st 2023 07:00:28 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from recsys.repo.base import Repo
from recsys.workspace.workspace import Workspace
from recsys.assets.data import DataAsset
from recsys.data.repo import DataAssetRepo
from recsys.model.repo import ModelAssetRepo
from recsys.experiment.repo import ExperimentAssetRepo
from recsys.data.rating import RatingsDataset
from recsys.assets.model import ModelAsset
from recsys.assets.experiment import ExperimentAsset
from recsys.persistence.io import IOService
from recsys.operator.data.sampling import UserRandomSampling
from recsys.exceptions.workspace import WorkspaceException
from recsys.persistence.odb import ObjectDB


# ------------------------------------------------------------------------------------------------ #
@dataclass
class WorkspaceConfig:  # pragma no cover
    name: str
    description: str
    studio: str
    dataset_name: str
    dataset_description: str
    dataset_filepath: str  # Filepath to the original dataset from which a sample will be taken.
    dataset_size: float  # The proportion of the original dataset to include


# ------------------------------------------------------------------------------------------------ #
class AbstractWorkspaceFactory(ABC):
    def __Init__(self, config: WorkspaceConfig) -> None:
        self._config = config
        self._location = None
        self._workspace = None
        self._config = None

    @abstractmethod
    def create_workspace(self) -> None:
        """Creates workspace"""

    @abstractmethod
    def create_dataset_repo(self) -> None:
        """Creates dataset repo"""

    @abstractmethod
    def create_model_repo(self) -> None:
        """Creates model repo"""

    @abstractmethod
    def create_experiment_repo(self) -> None:
        """Creates experiment repo"""

    @abstractmethod
    def create_dataset(self) -> Repo:
        """Creates seeding dataset"""


# ------------------------------------------------------------------------------------------------ #
class WorkspaceFactory(AbstractWorkspaceFactory):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__(config=config)

    def create_workspace(self) -> None:
        """Method that controls the construction process."""
        self._set_location()
        self._workspace = Workspace(
            name=self._config.name, description=self._config.description, location=self._location
        )

    def create_data_repo(self) -> None:
        filepath = os.path.join(self._location, "dataset", "dataset.db")
        db = ObjectDB(filepath=filepath)
        self._workspace.data_repo = DataAssetRepo(database=db, asset_type=DataAsset)

    def create_model_repo(self) -> None:
        """Creates the model repository."""
        filepath = os.path.join(self._location, "model", "model.db")
        db = ObjectDB(filepath=filepath)
        self._workspace.model_repo = ModelAssetRepo(database=db, asset_type=ModelAsset)

    def create_experiment_repo(self) -> None:
        """Creates the experiment repository."""
        filepath = os.path.join(self._location, "experiment", "experiment.db")
        db = ObjectDB(filepath=filepath)
        self._workspace.experiment_repo = ExperimentAssetRepo(
            database=db, asset_type=ExperimentAsset
        )

    def create_dataset(self) -> None:
        """Builds the dataset and adds it to the Dataset repo."""
        data = IOService.read(self._config.dataset)
        sampler = UserRandomSampling(frac=self._config.size)
        sample = sampler.run(data)
        dataset = RatingsDataset(
            name=self._config.dataset_name,
            description=self._config.dataset_description,
            data=sample,
        )
        self._workspace.data_repo.add(asset=dataset)

    def check_workspace(self) -> bool:
        """Confirms the workspace has all components. Throws exception if missing component"""
        if self._workspace.dataset is None:
            msg = f"Workspace {self._config.name} missing dataset repository"
            self._log_raise(msg)
        elif self._workspace.model is None:
            msg = f"Workspace {self._config.name} missing model repository"
            self._log_raise(msg)
        elif self._workspace.experiment is None:
            msg = f"Workspace {self._config.name} missing experiment repository"
            self._log_raise(msg)
        elif self._workspace.dataset.exists(key=self._config.dataset_name) is False:
            msg = f"Workspace {self._config.name} dataset repository is missing the dataset."
            self._log_raise(msg)
        return True

    def _set_location(self) -> None:
        self._location = os.path.join(self._studio, self._config.name)

    def _log_raise(self, msg: str) -> None:
        """Logs and raises WorkspaceException with the specified message."""
        self._logger.error(msg)
        raise WorkspaceException(msg)
