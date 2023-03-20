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
# Modified   : Monday March 20th 2023 03:01:56 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Persistence Base Class"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """Based class for all assets.

    Args:
        name (str): Name of the asset instance
        atype type[Asset]: A class type
        desc (str): desc of the asset
    """

    def __init__(self, name: str, desc: str) -> None:
        self._name = name
        self._desc = desc

    @property
    def id(self) -> pd.DataFrame:
        """Returns name, desc, and type in DataFrame format for registration"""
        d = {"name": self._name, "type": self.__class__.__name__, "description": self._desc}
        df = pd.DataFrame(data=d, index=[0])
        return df

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        "Returns the asset desc"
        return self._desc


# ------------------------------------------------------------------------------------------------ #
class AssetRepoABC(ABC):
    """Abstract base class for asset repositories.

    Each subclass maintains the persistence of a single asset type.

    Note: Each subclass must override the constructor designating location and tablename.

    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

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
