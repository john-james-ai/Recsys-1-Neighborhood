#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/asset/centre.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 20th 2023 03:02:12 am                                                  #
# Modified   : Monday March 20th 2023 04:06:39 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Asset Centre Module"""
import logging


from recsys.asset.base import Asset
from recsys.asset.repo import AssetRepo


# ------------------------------------------------------------------------------------------------ #
class AssetCentre:
    """Provides runtime read write access to Asset repositories"""

    def __init__(
        self,
        datasource: AssetRepo,
        dataset: AssetRepo,
        operator: AssetRepo,
        algorithm: AssetRepo,
        model: AssetRepo,
        pipeline: AssetRepo,
    ) -> None:
        self._repos = {
            "datasource": datasource,
            "dataset": dataset,
            "operator": operator,
            "algorithm": algorithm,
            "model": model,
            "pipeline": pipeline,
        }
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def add_datasource(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("datasource")
        repo.add(asset=asset)

    def get_datasource(self, name: str) -> Asset:
        """Obtains a named datasource from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("datasource")
        return repo.get(name=name)

    def add_dataset(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("dataset")
        repo.add(asset=asset)

    def get_dataset(self, name: str) -> Asset:
        """Obtains a named dataset from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("dataset")
        return repo.get(name=name)

    def add_operator(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("operator")
        repo.add(asset=asset)

    def get_operator(self, name: str) -> Asset:
        """Obtains a named operator from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("operator")
        return repo.get(name=name)

    def add_algorithm(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("algorithm")
        repo.add(asset=asset)

    def get_algorithm(self, name: str) -> Asset:
        """Obtains a named algorithm from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("algorithm")
        return repo.get(name=name)

    def add_model(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("model")
        repo.add(asset=asset)

    def get_model(self, name: str) -> Asset:
        """Obtains a named model from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("model")
        return repo.get(name=name)

    def add_pipeline(self, asset: Asset) -> None:
        """Adds an asset to the appropriate repository.

        Args:
            asset (Asset): The asset to be persisted.
        """
        repo = self._repos.get("pipeline")
        repo.add(asset=asset)

    def get_pipeline(self, name: str) -> Asset:
        """Obtains a named pipeline from the repository.

        Args:
            name (str): The name of the asset.
        """
        repo = self._repos.get("pipeline")
        return repo.get(name=name)
