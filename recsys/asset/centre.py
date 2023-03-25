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
# Modified   : Monday March 20th 2023 11:28:18 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Datasource Centre Module"""
import os
import shutil
import logging

from sqlalchemy import text, exc
import pandas as pd

from recsys.services.io import IOService
from recsys.asset.base import AssetCentreABC, Asset
from recsys.persistence.database import Database


# ------------------------------------------------------------------------------------------------ #
class AssetCentre(AssetCentreABC):
    """Repository for data, model, and related assets.

    Args:
        database (Database): A Database object containing the registry
        directory (str): Persistence location for all assets.
        tablename (str): The name of the table containing the registry.
        io (type[IOService]): Read write capability for files.
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
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def add(self, asset: Asset) -> Asset:
        """Adds an asset to the repository and returns it.

        Args:
            asset (Asset): The asset instance.
        Raises: FileExistsError if file already exists.
        """
        asset = self._set_filepath(asset)
        try:
            if self.exists(name=asset.name, asset_type=asset.__class__.__name__):
                msg = f"An asset of type {asset.__class__.__name__} named {asset.name} already exists. Change the name or replace the asset."
                self._logger.error(msg)
                raise FileExistsError(msg)
        except exc.OperationalError:  # If first add, database won't exist at time of existence check.
            pass

        self._check_in(asset)
        self._save(asset)
        return asset

    def get(self, name: str, asset_type: str) -> Asset:
        """Gets the asset with the designated name.

        Args:
            name (str): The name of the asset
            asset_type (str): The type or class name of the asset.

        """
        filepath = self._get_filepath(name=name, asset_type=asset_type)
        try:
            return self._load(filepath=filepath)
        except AttributeError:
            msg = f"Attribute named {name} of type{asset_type} does not exist."
            self._logger.error(msg)
            raise FileNotFoundError(msg)

    def remove(self, name: str, asset_type: str) -> None:
        """Removes the asset with the designated name from storage and registry.

        Args:
            name (str): The name of the asset
            asset_type (str): The type or class name of the asset.

        """
        filepath = self._get_filepath(name=name, asset_type=asset_type)
        self._check_out(name=name, asset_type=asset_type)
        self._delete(filepath=filepath)

    def replace(self, asset: Asset) -> None:
        """Replaces an asset in registry and storage.

        Args:
            asset (Asset): Asset to replace.
        Raises: FileNotFound if asset does not exist.
        """
        asset = self._set_filepath(asset)

        if not self.exists(name=asset.name, asset_type=asset.__class__.__name__):
            msg = f"An asset of type {asset.__class__.__name__} named {asset.name} does not exists. Try add method instead."
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        self._check_out(name=asset.name, asset_type=asset.__class__.__name__)
        self._check_in(asset=asset)
        self._save(asset)

    def exists(self, name: str, asset_type: str) -> bool:
        """Determines if an asset of the designated name and type exists.

        Args:
            name (str): Name of the asset.
            asset_type (str): The type of asset, i.e. class name.
        """
        query = text(
            f"SELECT EXISTS(SELECT 1 FROM {self._tablename} WHERE name='{name}' AND type='{asset_type}');"
        )
        exists = pd.read_sql(query, con=self._cxn).values[0][0]
        self._logger.debug(exists)
        return exists == 1

    def show(self) -> None:
        """Prints the registry to screen."""
        df = self._get_registry()
        df = df.drop(["index"], axis=1, errors="ignore")  # Kluge to dump extra index column.
        print(df)

    def reset(self, confirm: bool = True) -> None:
        """Resets the asset registry and PURGES the asset repository."""
        if confirm:
            confirmation = input(
                "This will permanently delete the entire repository. Are you SURE? [y/n]: "
            )
            if "y" in confirmation.lower():
                self._reset_registry()
                self._purge_assets()
        else:  # pragma: no cover
            self._reset_registry()
            self._purge_assets()

    def _check_in(self, asset: Asset) -> None:
        """Adds an asset to the registry."""
        df = self._get_entry(asset)
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

    def _check_out(self, name: str, asset_type: str) -> None:
        """Removes an asset from the registry by name.

        Args:
            name (str): Name of the asset.
            asset_type (str): The type of asset, i.e. class name.
        """
        query = text(f"DELETE FROM {self._tablename} WHERE name='{name}' AND type='{asset_type}';")
        self._cxn.execute(query)

    def _get_entry(self, asset: Asset) -> pd.DataFrame:
        """Extracts information from Asset and formats registration entry

        Args:
            asset (Asset): The asset to be persisted.
        """
        d = {
            "name": asset.name,
            "type": asset.__class__.__name__,
            "description": asset.desc,
            "filepath": asset.filepath,
        }
        df = pd.DataFrame(data=d, index=[0])
        return df

    def _get_filepath(self, name: str, asset_type: str) -> str:
        """Obtains the filepath for the name and asset type from the registry.

        Args:
           name (str): Name of the asset.
           asset_type (str): The type or class name of the asset.
        """
        query = text(
            f"SELECT filepath FROM {self._tablename} WHERE name='{name}' AND type='{asset_type}';"
        )
        try:
            return pd.read_sql(query, con=self._cxn).values[0][0]
        except IndexError:
            return False

    def _set_filepath(self, asset: Asset) -> Asset:
        """Constructs a filepath and sets the filepath attribute on the asset

        Args:
            asset (Asset): An asset object.
        """
        if asset.filepath is None:
            filename = asset.__class__.__name__ + "_" + asset.name + ".pkl"
            filepath = os.path.join(self._directory, filename)
            asset.filepath = filepath
        return asset

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
        except exc.OperationalError:  # pragma: no cover
            pass

    def _save(self, asset: Asset) -> None:
        """Persists the asset to storage

        Args:
            asset (Asset): The asset to persist.

        """
        self._io.write(filepath=asset.filepath, data=asset)

    def _load(self, filepath: str) -> Asset:
        """Returns the asset with the designated name from storage.

        Args:
            filepath (str): The filepath of the asset.
        """

        return self._io.read(filepath=filepath)

    def _delete(self, filepath: str) -> Asset:
        """Deletes the asset from storage (if it exists).

        Args:
            filepath (str): The filepath for the asset
        """
        if os.path.exists(filepath):
            os.remove(filepath)

    def _purge_assets(self) -> None:
        """Delete the directory containing assets."""
        shutil.rmtree(self._directory, ignore_errors=True)
