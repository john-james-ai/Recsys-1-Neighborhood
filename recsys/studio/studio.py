#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/studio/studio.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 11:01:00 pm                                               #
# Modified   : Wednesday March 1st 2023 07:09:56 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
import os
from dotenv import load_dotenv

from recsys.repo.base import Repo
from recsys.studio.workspace import Workspace, WorkspaceConfig, Workspacebuilder


# ------------------------------------------------------------------------------------------------ #
class Studio:
    def __init__(self, name, repo: Repo) -> None:
        self._name = name
        self._repo = repo

    def create_workspace(self, config: WorkspaceConfig) -> Workspace:
        builder = Workspacebuilder(studio=self._name)
        builder.build_workspace(config)
        builder.add_dataset_repo()
        builder.add_model_repo()
        builder.add_experiment_repo()
        builder.add_dataset()
        builder.check_workspace()
        return builder.workspace

    def add_workspace(self, workspace: Workspace) -> None:
        self._repo.add(workspace=workspace)

    def drop_workspace(self, name) -> None:
        """Drops the workspace"""
        self._repo.remove(key=name)

    def get_current_workspace(self) -> Workspace:
        load_dotenv()
        name = os.getenv("WORKSPACE")
        return self._repo.get(key=name)

    def get_workspace(self, name: str) -> Workspace:
        """Returns the workspace with the designated name"""
        return self._repo.get(key=name)

    def info(self) -> None:
        self._repo.info()
