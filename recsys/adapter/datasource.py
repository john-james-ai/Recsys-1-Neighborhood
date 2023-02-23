#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/adapter/datasource.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:25:30 am                                            #
# Modified   : Wednesday February 22nd 2023 04:25:27 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Datasource Module"""
from dataclasses import dataclass

from recsys.adapter.base import SQL
from recsys.dal.dto import DatasourceDTO
from recsys.adapter.base import Adapter

# ================================================================================================ #
#                                       DATASOURCE                                                 #
# ================================================================================================ #


# ------------------------------------------------------------------------------------------------ #
#                                          DDL                                                     #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class CreateDatasourceTable(SQL):
    name: str = "datasource"
    sql: str = """id INTEGER  PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL, author TEXT NOT NULL, publisher TEXT NOT NULL, published TEXT , version TEXT , website TEXT NOT NULL, uri TEXT NOT NULL, doi TEXT , email TEXT , created TEXT DEFAULT CURRENT_TIMESTAMP;"""
    args: tuple = ()
    description: str = "Created the datasource table."


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DropDatasourceTable(SQL):
    name: str = "datasource"
    sql: str = """DROP TABLE IF EXISTS datasource;"""
    args: tuple = ()
    description: str = "Dropped the datasource table."


# ------------------------------------------------------------------------------------------------ #


@dataclass
class DatasourceTableExists(SQL):
    name: str = "datasource"
    sql: str = None
    args: tuple = ()
    description: str = "Checked existence of datasource table."

    def __post_init__(self) -> None:
        self.sql = """SELECT COUNT(TABLE_NAME) FROM information_schema.TABLES WHERE TABLE_NAME = 'datasource';"""


# ------------------------------------------------------------------------------------------------ #
#                                          DML                                                     #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class InsertDatasource(SQL):
    dto: DatasourceDTO
    name: str = "datasource"
    sql: str = """INSERT INTO datasource (name, type, title, description, author, publisher, published, version, website, uri, doi, email)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.name,
            self.dto.type,
            self.dto.title,
            self.dto.description,
            self.dto.author,
            self.dto.publisher,
            self.dto.published,
            self.dto.version,
            self.dto.website,
            self.dto.uri,
            self.dto.doi,
            self.dto.email,
        )


# ------------------------------------------------------------------------------------------------ #


@dataclass
class UpdateDatasource(SQL):
    dto: DatasourceDTO
    name: str = "datasource"
    sql: str = """UPDATE datasource SET name = ?, type = ?, title = ?, description = ?, author = ?, publisher = ?, published = ?, version = ?, website = ?, uri = ?, doi = ?, email = ? WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.name,
            self.dto.type,
            self.dto.title,
            self.dto.description,
            self.dto.author,
            self.dto.publisher,
            self.dto.published,
            self.dto.version,
            self.dto.website,
            self.dto.uri,
            self.dto.doi,
            self.dto.email,
            self.dto.id,
        )


# ------------------------------------------------------------------------------------------------ #


@dataclass
class ReadDatasource(SQL):
    id: int
    name: str = "datasource"
    sql: str = """SELECT * FROM datasource WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #


@dataclass
class ReadAllDatasources(SQL):
    name: str = "datasource"
    sql: str = """SELECT * FROM datasource;"""
    args: tuple = ()


# ------------------------------------------------------------------------------------------------ #


@dataclass
class DatasourceExists(SQL):
    id: int
    name: str = "datasource"
    sql: str = """SELECT EXISTS(SELECT 1 FROM datasource WHERE id = ? LIMIT 1);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DeleteDatasource(SQL):
    id: int
    name: str = "datasource"
    sql: str = """DELETE FROM datasource WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Respond:
    row: list

    def as_dto(self) -> None:
        return DatasourceDTO(
            id=self.row[0],
            name=self.row[1],
            type=self.row[2],
            title=self.row[3],
            description=self.row[4],
            author=self.row[5],
            publisher=self.row[6],
            published=self.row[7],
            version=self.row[8],
            website=self.row[9],
            uri=self.row[10],
            doi=self.row[11],
            email=self.row[12],
            created=self.row[13],
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasourceAdapter(Adapter):
    create: SQL = CreateDatasourceTable()
    drop: SQL = DropDatasourceTable()
    table_exists: SQL = DatasourceTableExists()
    insert: type[SQL] = InsertDatasource
    update: type[SQL] = UpdateDatasource
    read: type[SQL] = ReadDatasource
    read_all: type[SQL] = ReadAllDatasources
    row_exists: type[SQL] = DatasourceExists
    delete: type[SQL] = DeleteDatasource
    respond: DatasourceDTO = Respond
