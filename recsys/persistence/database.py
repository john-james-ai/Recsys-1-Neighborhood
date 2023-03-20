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
# Modified   : Sunday March 19th 2023 05:17:35 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from sqlalchemy import create_engine
from sqlalchemy import engine


# ------------------------------------------------------------------------------------------------ #
class Database:
    __location = "sqlite:///assets/registry.db"

    def __init__(self) -> None:
        self._engine = create_engine(Database.__location)
        self._cxn = engine.connect()

    @property
    def connection(self) -> engine.connect:
        return self._cxn
