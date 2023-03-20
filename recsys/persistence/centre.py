#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/persistence/centre.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday March 19th 2023 04:20:33 pm                                                  #
# Modified   : Sunday March 19th 2023 06:53:03 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Datasource Centre Module"""
import os

from sqlalchemy import engine
import pandas as pd

from recsys.services.io import IOService
from recsys.persistence.base import AssetCentreABC, Asset


# ------------------------------------------------------------------------------------------------ #
class AssetCentre(AssetCentreABC):
    """Abstract base class for asset repositories.

    Each subclass maintains the persistence of a single asset type.

    Note: Each subclass must override the constructor designating location and tablename.

    """

    def __init__(
        self, cxn: engine.connection, location: str, tablename: str, io: IOService = IOService
    ) -> None:
        self._cxn = cxn
        self._io = io
        self._location = location
        self._tablename = tablename

    def add(self, asset: Asset) -> None:
        """Adds an asset to the repository

        Args:
            asset (Asset): The asset instance.
        Raises: FileExistsError if file already exists.
        """
        if self.exists(name=asset.name):
            msg = (
                f"An asset named {asset.name} already exists. Change the name or replace the asset."
            )
            self._logger.error(msg)
            raise FileExistsError(msg)

        self._register(asset)
        self._save(asset)

    def get(self, name: str) -> Asset:
        """Gets the asset with the designated name."""
        return self._load(name=name)

    def remove(self, name: str) -> None:
        """Removes the asset with the designated name from storage and registry."""
        self._unregister(name=name)
        self._delete(name=name)

    def replace(self, asset: Asset) -> None:
        """Replaces an asset in registry and storage.

        Args:
            asset (Asset): Asset to replace.
        Raises: FileNotFound if asset does not exist.
        """

        if not self.exists(name=asset.name):
            msg = f"This asset named, {asset.name}, does not exist. Try the add method."
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        self._unregister(name=asset.name)
        self._register(asset=asset)
        self._save(asset)

    def exists(self, name: str) -> bool:
        """Determines if an asset of the designated name exists.

        Args:
            name (str): Name of the asset.
        """
        df = self._get_registry()
        asset_df = df[df["name"] == name]
        return asset_df.shape[0] > 0

    def _register(self, asset: Asset) -> None:
        """Adds an asset to the registry."""
        df = asset.to_df()
        df.to_sql(name=self._tablename, con=self._cxn, if_exists="append")

    def _unregister(self, name: str) -> None:
        """Removes an asset from the registry by name.

        Args:
            name (str): Name of the asset.
        """
        df = self._get_registry()
        df = df[df["name"] != name]
        self._put_registry(df=df)

    def _get_registry(self) -> pd.DataFrame:
        """Returns the registry dataframe."""
        return pd.read_sql("SELECT * FROM datasource")

    def _put_registry(self, df: pd.DataFrame) -> None:
        """Saves the registry, overwriting the existing registry. Used when deleting assets.

        Note: This REPLACES the entire registry.

        Args:
            df (pd.DataFrame): The registry in DataFrame format.
        """
        df.to_sql(name=self._tablename, con=self._cxn, if_exists="replace")

    def _save(self, asset: Asset) -> None:
        """Persists the asset to storage

        Args:
            asset (Asset): The asset to persist.

        """
        filename = asset.name + ".pkl"
        filepath = os.path.join(self._location, filename)
        self._io.write(filepath=filepath, data=asset)

    def _load(self, name: str) -> Asset:
        """Returns the asset with the designated name from storage.

        Args:
            name (str): The name for the asset
        """
        filename = name + ".pkl"
        filepath = os.path.join(self._location, filename)
        return self._io.read(filepath=filepath)

    def _delete(self, name: str) -> Asset:
        """Deletes the asset from storage.

        Args:
            name (str): The name for the asset
        """
        filename = name + ".pkl"
        filepath = os.path.join(self._location, filename)
        os.remove(filepath)
