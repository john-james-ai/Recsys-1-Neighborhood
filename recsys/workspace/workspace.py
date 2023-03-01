#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workspace/workspace.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 03:35:03 am                                                #
# Modified   : Wednesday March 1st 2023 07:01:18 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys import Asset
from recsys.repo.base import Repo


# ------------------------------------------------------------------------------------------------ #
class Workspace(Asset):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self._data_repo = None
        self._model_repo = None
        self._experiment_repo = None

    @property
    def data_repo(self) -> Repo:
        return self._data_repo

    @data_repo.setter
    def data_repo(self, repo: Repo) -> None:
        self._data_repo = repo

    @property
    def model_repo(self) -> Repo:
        return self._model_repo

    @model_repo.setter
    def model_repo(self, repo: Repo) -> None:
        self._model_repo = repo

    @property
    def experiment_repo(self) -> Repo:
        return self._experiment_repo

    @experiment_repo.setter
    def experiment_repo(self, repo: Repo) -> None:
        self._experiment_repo = repo
