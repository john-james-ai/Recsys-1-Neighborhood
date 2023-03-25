#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/persistence/database.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday March 19th 2023 04:32:25 pm                                                  #
# Modified   : Monday March 20th 2023 10:43:21 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os

from sqlalchemy import create_engine, engine


# ------------------------------------------------------------------------------------------------ #
class Database:
    """Sqlite database using the SQLAlchemy ORM

    Args:
        filepath (str): The path to the sqlite3 database file
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    @property
    def engine(self) -> engine:
        """Returns an SQLAlchemy engine."""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        engine = create_engine(self._filepath)
        return engine
