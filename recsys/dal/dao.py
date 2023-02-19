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
# Created    : Sunday February 19th 2023 04:15:04 am                                               #
# Modified   : Sunday February 19th 2023 04:20:54 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.data.rdbms import SQLite
from recsys.dal.sql import ObjectDML


class DAO:
    """Data Access Object to Database

    Args:
        db (Database): Database object
        dml (SQL): Data manipulation language class
    """

    def __init__(self, db: SQLite, dml: ObjectDML) -> None:
        self._db = db()
        self._dml = dml()
