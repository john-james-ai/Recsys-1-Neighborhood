#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dal/sql.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 18th 2023 05:48:28 pm                                             #
# Modified   : Saturday February 18th 2023 08:48:48 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

from recsys.dal.base import DTO, SQL


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CreateObjectTable(SQL):
    name: str = "object"
    sql: str = """CREATE TABLE IF NOT EXISTS object ( id SMALLINT NOT NULL, type TEXT NOT NULL, name TEXT NOT NULL, description TEXT, memory REAL NOT NULL DEFAULT 0, cost REAL NOT NULL DEFAULT 0, created DATETIME DEFAULT CURRENT_TIMESTAMP, updated DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (id ); """
    args: tuple = ()
    description: str = "Create Object Table"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DropObjectTable(SQL):
    name: str = "object"
    sql: str = """DROP TABLE IF EXISTS object;"""
    args: tuple = ()
    description: str = "Dropped the object table"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class InsertObject(SQL):
    dto: DTO
    sql: str = """INSERT INTO object (type, name, description, memory, cost) VALUES (?,?,?,?,?);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.type,
            self.dto.name,
            self.dto.description,
            self.dto.memory,
            self.dto.cost,
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class UpdateObject(SQL):
    dto: DTO
    sql: str = """UPDATE object SET type = ?, name = ?, description = ?, memory = ?, cost = ?  WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.type,
            self.dto.name,
            self.dto.description,
            self.dto.memory,
            self.dto.cost,
            self.dto.id,
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SelectObject(SQL):
    id: int
    sql: str = """SELECT * FROM object WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SelectObjectByName(SQL):
    name: str
    sql: str = """SELECT * FROM object WHERE name = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.name,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SelectAllObjects(SQL):
    sql: str = """SELECT * FROM object;"""
    args: tuple = ()


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ObjectExists(SQL):
    id: int
    sql: str = """SELECT EXISTS(SELECT 1 FROM object WHERE id = ? LIMIT 1);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DeleteObject(SQL):
    id: int
    sql: str = """DELETE FROM object WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ObjectDML:
    create: type[SQL] = CreateObjectTable
    drop: type[SQL] = DropObjectTable
    insert: type[SQL] = InsertObject
    update: type[SQL] = UpdateObject
    select: type[SQL] = SelectObject
    select_by_name: type[SQL] = SelectObjectByName
    select_all: type[SQL] = SelectAllObjects
    exists: type[SQL] = ObjectExists
    delete: type[SQL] = DeleteObject
