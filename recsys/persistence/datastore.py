#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/datastore.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 11:01:00 pm                                               #
# Modified   : Tuesday February 28th 2023 01:39:54 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

from recsys.persistence.repo import Repo


# ------------------------------------------------------------------------------------------------ #
class DataStore:

    __home = "devstudio"

    def __init__(
        self, workspace: str, dataset: Repo, experiment: Repo, model: Repo, datasize: float
    ) -> None:
        self._workspace = workspace
        self._dataset = dataset
        self._experiment = experiment
        self._model = model
        self._datasize = datasize
        self._size = 0

    @property
    def dataset(self) -> Repo:
        return self._dataset

    @property
    def experiment(self) -> Repo:
        return self._experiment

    @property
    def model(self) -> Repo:
        return self.model
