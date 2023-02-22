#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dal/dao.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 03:40:35 am                                            #
# Modified   : Wednesday February 22nd 2023 03:05:55 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Access Object Module"""
import logging

import pandas as pd

from recsys.dal.base import DAOBase
from recsys.persistence.base import Database
from recsys.dal.base import DTO
from recsys.adapter.dataset import DatasetAdapter


# ------------------------------------------------------------------------------------------------ #
class DAO(DAOBase):
    """Generic Data Access Object

    Args:
        dml (DML): A Data Manipulation Language
        database (Database): A relational database object.
    """

    def __init__(self, adapter: DatasetAdapter, database: Database) -> None:
        super().__init__()
        self._adapter = adapter
        self._database = database
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def create(self, dto: DTO) -> DTO:
        """Adds an entity in the form of a data transfer object to the database.
        Args:
            dto (DTO): An entity data transfer object.

        """
        cmd = self._adapter.insert(dto)
        dto.id = self._database.insert(cmd.sql, cmd.args)
        self._database.commit()
        msg = f"{self.__class__.__name__} inserted {self._adapter.insert.name}.{dto.id} - {dto.name} into database at {id(self._database)}."
        self._logger.debug(msg)
        return dto

    def read(self, id: int) -> DTO:
        """Obtains an entity DTO with the designated id.
        Args:
            id (int): The id for the entity.
        Returns a DTO
        """
        cmd = self._adapter.read(id)
        row = self._database.select(cmd.sql, cmd.args)
        return self._adapter.respond(row).as_dto()

    def read_all(self) -> pd.DataFrame:
        """Returns a DataFrame containing all data in the table."""
        cmd = self._adapter.read_all()
        return self._database.select_all(cmd.sql, cmd.args)

    def update(self, dto: DTO) -> int:
        """Performs an update to an existing entity DTO
        Args:
            dto (DTO): Data Transfer Object
        Returns number of rows effected.
        """
        rows_affected = None
        if self.exists(dto.id):
            cmd = self._adapter.update(dto)
            rows_affected = self._database.update(cmd.sql, cmd.args)
            self._database.commit()

        else:
            msg = f"{self.__class__.__name__} was unable to update {self._adapter.update.name}.{dto.id}. Not found in the database. Try insert instead."
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        return rows_affected

    def exists(self, id: int) -> bool:
        """Returns True if the entity with id exists in the database.
        Args:
            id (int): id for the entity
        """
        cmd = self._adapter.row_exists(id)
        result = self._database.exists(cmd.sql, cmd.args)
        return result

    def delete(self, id: int, persist=True) -> None:
        """Deletes a Entity from the registry, given an id.
        Args:
            id (int): The id for the entity to delete.
        """
        if self.exists(id):
            cmd = self._adapter.delete(id)
            self._database.delete(cmd.sql, cmd.args)
            self._database.commit()
        else:
            msg = f"{self.__class__.__name__}  was unable to delete {self._adapter.delete.name}.{id}. Not found in the database."
            self._logger.error(msg)
            raise FileNotFoundError(msg)
