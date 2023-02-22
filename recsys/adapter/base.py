#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/adapter/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 01:03:35 pm                                            #
# Modified   : Wednesday February 22nd 2023 01:40:51 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Adapter Module"""
from dataclasses import dataclass
from abc import ABC

from recsys.dal.dto import DatasetDTO


# ------------------------------------------------------------------------------------------------ #
#                                  SQL COMMAND ABC                                                 #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class SQL(ABC):  # pragma: no cover
    """Base class for SQL Command Objects."""


# ------------------------------------------------------------------------------------------------ #
#                                 BASE ADAPTER CLASS                                               #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Adapter(ABC):  # pragma: no cover
    """Defines interface of messages that the adapter supports."""

    create: type[SQL] = None
    drop: type[SQL] = None
    table_exists: type[SQL] = None
    insert: type[SQL] = None
    update: type[SQL] = None
    read: type[SQL] = None
    read_all: type[SQL] = None
    row_exists: type[SQL] = None
    delete: type[SQL] = None
    respond: DatasetDTO = None
