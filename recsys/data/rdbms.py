#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/rdbms.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday February 20th 2023 09:10:07 pm                                               #
# Modified   : Monday February 20th 2023 09:44:33 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.data.base import Database
import os
import sqlite3
from dotenv import load_dotenv


# ------------------------------------------------------------------------------------------------ #
#                                        DATABASE                                                  #
# ------------------------------------------------------------------------------------------------ #
class SQLite(Database):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self._connection = None
        self._location = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        if self._connection is not None:
            if isinstance(exc_value, Exception):
                self._connection.rollback()
            else:
                self._connection.commit()
            self._connection.close()

    def connect(self) -> None:
        """Connects to the database."""
        self._location = self._config.get("location")
        self._connection = sqlite3.connect(self._location)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()
        self._connection = None

    def command(self, sql: str, args: tuple = None) -> sqlite3.connection.cursor:
        """Executes a command on the database and returns a cursor object."""

        cursor = self._connection.cursor
        try:
            cursor.execute(sql, args)
        except sqlite3.Error as err:  # pragma: no cover
            self._logger.error(err)
            raise sqlite3.Error()

        return cursor

    def insert(self, sql: str, args: tuple = None) -> int:
        """Inserts data into a table and returns the last row id."""
        cursor = self.command(sql, args)
        id = cursor.lastrowid
        cursor.close()
        return id

    def select(self, sql: str, args: tuple = None) -> tuple:
        """Performs a select command returning a single instance or row."""
        row = None
        cursor = self.command(sql, args)
        row = cursor.fetchone()
        cursor.close()
        return row

    def select_all(self, sql: str, args: tuple = None) -> list:
        """Performs a select command returning multiple instances or rows."""
        rows = []
        cursor = self.command(sql, args)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def update(self, sql: str, args: tuple = None) -> None:
        """Performs an update on existing data in the database."""
        cursor = self.command(sql, args)
        rowcount = cursor.rowcount
        cursor.close()
        return rowcount

    def delete(self, sql: str, args: tuple = None) -> None:
        """Deletes existing data."""
        cursor = self.command(sql, args)
        cursor.close()
