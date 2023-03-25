#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/asset/base.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday March 19th 2023 04:11:20 pm                                                  #
# Modified   : Monday March 20th 2023 09:48:01 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Persistence Base Class"""
from __future__ import annotations
from abc import ABC, abstractmethod


# ------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """Based class for all assets.

    Args:
        name (str): Name of the asset instance
        desc (str): desc of the asset
    """

    def __init__(self) -> None:
        self._filepath = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the asset name"""

    @property
    @abstractmethod
    def desc(self) -> str:
        "Returns the asset description"

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, filepath: str) -> None:
        if self._filepath is None:
            self._filepath = filepath
        else:
            msg = (
                "Attempt to change the filepath failed. The filepath parameter may not be modified."
            )
            self._logger.error(msg)
            raise RuntimeError(msg)


# ------------------------------------------------------------------------------------------------ #
class AssetCentreABC(ABC):
    """Abstract base class for asset repositories.

    Each subclass maintains the persistence of a single asset type.

    Note: Each subclass must override the constructor designating location and tablename.

    """

    @abstractmethod
    def add(self, asset: Asset) -> None:
        """Adds an asset to the repository

        Args:
            asset (Asset): The asset instance.
        Raises: FileExistsError if file already exists.
        """

    @abstractmethod
    def get(self, name: str) -> Asset:
        """Gets the asset with the designated name."""

    @abstractmethod
    def remove(self, name: str) -> None:
        """Removes the asset with the designated name from storage and registry."""

    @abstractmethod
    def replace(self, asset: Asset) -> None:
        """Replaces an asset in registry and storage.

        Args:
            asset (Asset): Asset to replace.
        Raises: FileNotFound if asset does not exist.
        """

    @abstractmethod
    def show(self) -> None:
        """Prints the registry."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Determines if an asset of the designated name exists.

        Args:
            name (str): Name of the asset.
        """
