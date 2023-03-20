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
# Created    : Sunday March 19th 2023 04:20:33 pm                                                  #
# Modified   : Monday March 20th 2023 02:25:18 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Datasource Centre Module"""
import os
import shutil

from sqlalchemy import text, exc
import pandas as pd

from recsys.services.io import IOService
from recsys.asset.base import AssetCentreABC, Asset
from recsys.persistence.database import Database


# ------------------------------------------------------------------------------------------------ #
class AssetCentre(AssetCentreABC):
    """Abstract base class for asset repositories.

    Each subclass maintains the persistence of a single asset type.

    Note: Each subclass must override the constructor designating directory and tablename.

    """

    def __init__(
        self,
        database: Database,
        directory: str,
        tablename: str,
        io: IOService = IOService,
    ) -> None:
        super().__init__()
        self._engine = database.engine
        self._cxn = database.engine.connect()
        self._io = io
        self._directory = directory
        self._tablename = tablename

    def add(self, asset: Asset) -> None:
        """Adds an asset to the repository

        Args:
            asset (Asset): The asset instance.
        Raises: FileExistsError if file already exists.
        """
        try:
            if self.exists(name=asset.name):
                msg = f"An asset named {asset.name} already exists. Change the name or replace the asset."
                self._logger.error(msg)
                raise FileExistsError(msg)
        except exc.OperationalError:  # If first add, database won't exist at time of existence check.
            pass

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

    def show(self) -> None:
        """Prints the registry to screen."""
        df = self._get_registry()
        df = df.drop(["index"], axis=1, errors="ignore")  # Kluge to dump extra index column.
        print(df)

    def reset(self) -> None:
        """Resets the asset registry and PURGES the asset repository."""
        self._reset_registry()
        self._purge_assets()

    def _register(self, asset: Asset) -> None:
        """Adds an asset to the registry."""
        df = asset.id
        self._logger.debug(df)
        try:
            df.to_sql(
                name=self._tablename,
                con=self._cxn,
                if_exists="append",
                index=False,
            )
        except Exception:  # pragma: no cover
            msg = f"Exception when attempting to add asset {asset.name} to the registry in the {self._tablename} table."
            self._logger.error(msg)
            raise

    def _unregister(self, name: str) -> None:
        """Removes an asset from the registry by name.

        Args:
            name (str): Name of the asset.
        """
        query = text(f"DELETE FROM {self._tablename} WHERE name='{name}';")
        self._cxn.execute(query)

    def _get_registry(self) -> pd.DataFrame:
        """Returns the registry dataframe."""
        try:
            query = text(f"SELECT * from {self._tablename};")
            return pd.read_sql(query, con=self._cxn)

        except Exception:  # pragma: no cover
            msg = (
                f"Exception when attempting to read the registry from the {self._tablename} table."
            )
            self._logger.error(msg)
            raise

    def _reset_registry(self) -> None:
        """Drops the registry table."""
        query = text(f"DROP TABLE {self._tablename};")
        try:
            self._cxn.execute(query)
        except exc.OperationalError:  # In case it doesn't exist.
            pass

    def _save(self, asset: Asset) -> None:
        """Persists the asset to storage

        Args:
            asset (Asset): The asset to persist.

        """
        filename = asset.name + ".pkl"
        filepath = os.path.join(self._directory, filename)
        self._io.write(filepath=filepath, data=asset)

    def _load(self, name: str) -> Asset:
        """Returns the asset with the designated name from storage.

        Args:
            name (str): The name for the asset
        """
        filename = name + ".pkl"
        filepath = os.path.join(self._directory, filename)
        return self._io.read(filepath=filepath)

    def _delete(self, name: str) -> Asset:
        """Deletes the asset from storage (if it exists).

        Args:
            name (str): The name for the asset
        """
        filename = name + ".pkl"
        filepath = os.path.join(self._directory, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    def _purge_assets(self) -> None:
        """Delete the directory containing assets."""
        shutil.rmtree(self._directory, ignore_errors=True)
