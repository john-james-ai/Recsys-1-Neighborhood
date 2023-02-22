#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/rdbms.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday February 20th 2023 09:10:07 pm                                               #
# Modified   : Wednesday February 22nd 2023 02:46:09 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.persistence.base import Database
import sqlite3
import pandas as pd


# ------------------------------------------------------------------------------------------------ #
#                                        DATABASE                                                  #
# ------------------------------------------------------------------------------------------------ #
class SQLite(Database):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self._location = config
        self._connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, ext_type, ext_value, traceback):
        if self._connection is not None:
            if isinstance(ext_value, Exception):
                self._connection.rollback()
                self._logger.error(f"\nExit Type: {ext_type}")
                self._logger.error(f"\nExit Value: {ext_value}")
                self._logger.error(f"\nTraceback: {traceback}")
            else:
                self._connection.commit()
            self._connection.close()

    def commit(self) -> None:
        self._connection.commit()

    def connect(self) -> None:
        """Connects to the database."""
        self._connection = sqlite3.connect(self._location)

    def close(self) -> None:
        """Closes the underlying database connection."""
        self._connection.close()
        self._connection = None

    def command(self, sql: str, args: tuple = None):
        """Executes a command on the database and returns a cursor object."""

        if not self._connection:
            self.connect()
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, args)
        except sqlite3.Error as err:  # pragma: no cover
            self._logger.error(err)
            raise sqlite3.Error()

        return cursor

    def create(self, sql: str, args: tuple = None) -> int:
        """Inserts data into a table and returns the last row id."""
        cursor = self.command(sql, args)
        cursor.close()

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
        return pd.read_sql_query(sql, con=self._connection, params=args)

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

    def exists(self, sql: str, args: tuple = None) -> bool:
        """Returns True if the data specified by the parameters exists. Returns False otherwise."""
        result = []
        cursor = self.command(sql, args)
        result = cursor.fetchone()
        cursor.close()
        try:
            return result[0] == 1
        except IndexError:  # pragma: no cover
            return False

    def drop(self, sql: str, args: tuple = None) -> None:
        """Drop a database or table."""
        cursor = self.command(sql, args)
        cursor.close()
