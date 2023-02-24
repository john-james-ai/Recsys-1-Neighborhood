#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/adapter/dataset.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:25:30 am                                            #
# Modified   : Wednesday February 22nd 2023 10:58:15 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Dataset Data Minipulation Language Module"""
from dataclasses import dataclass

from recsys.adapter.base import SQL
from recsys.dal.dto import DatasetDTO
from recsys.adapter.base import Adapter

# ================================================================================================ #
#                                        DATASET                                                   #
# ================================================================================================ #


# ------------------------------------------------------------------------------------------------ #
#                                          DDL                                                     #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class CreateDatasetTable(SQL):
    name: str = "dataset"
    sql: str = """CREATE TABLE IF NOT EXISTS dataset (id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, description TEXT , lab TEXT NOT NULL, stage TEXT NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, n_users INTEGER NOT NULL, n_items INTEGER NOT NULL, size INTEGER NOT NULL, matrix_size INTEGER NOT NULL, memory_mb REAL NOT NULL, cost INTEGER NOT NULL, sparsity REAL NOT NULL, density REAL NOT NULL, filepath TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);"""
    args: tuple = ()
    description: str = "Created the dataset table."


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DropDatasetTable(SQL):
    name: str = "dataset"
    sql: str = """DROP TABLE IF EXISTS dataset;"""
    args: tuple = ()
    description: str = "Dropped the dataset table."


# ------------------------------------------------------------------------------------------------ #


@dataclass
class DatasetTableExists(SQL):
    name: str = "dataset"
    sql: str = None
    args: tuple = ()
    description: str = "Checked existence of dataset table."

    def __post_init__(self) -> None:
        self.sql = """SELECT COUNT(TABLE_NAME) FROM information_schema.TABLES WHERE TABLE_NAME = 'dataset';"""


# ------------------------------------------------------------------------------------------------ #
#                                          DML                                                     #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class InsertDataset(SQL):
    dto: DatasetDTO
    name: str = "dataset"
    sql: str = """INSERT INTO dataset (
                    name,
                    type,
                    description,
                    lab,
                    stage,
                    rows,
                    cols,
                    n_users,
                    n_items,
                    size,
                    matrix_size,
                    memory_mb,
                    cost,
                    sparsity,
                    density,
                    filepath)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.name,
            self.dto.type,
            self.dto.description,
            self.dto.lab,
            self.dto.stage,
            self.dto.rows,
            self.dto.cols,
            self.dto.n_users,
            self.dto.n_items,
            self.dto.size,
            self.dto.matrix_size,
            self.dto.memory_mb,
            self.dto.cost,
            self.dto.sparsity,
            self.dto.density,
            self.dto.filepath,
        )


# ------------------------------------------------------------------------------------------------ #


@dataclass
class UpdateDataset(SQL):
    dto: DatasetDTO
    name: str = "dataset"
    sql: str = """UPDATE dataset SET name = ?,
                    type = ?,
                    description = ?,
                    lab = ?,
                    stage = ?,
                    rows = ?,
                    cols = ?,
                    n_users = ?,
                    n_items = ?,
                    size = ?,
                    matrix_size = ?,
                    memory_mb = ?,
                    cost = ?,
                    sparsity = ?,
                    density = ?,
                    filepath = ?
                    WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (
            self.dto.name,
            self.dto.type,
            self.dto.description,
            self.dto.lab,
            self.dto.stage,
            self.dto.rows,
            self.dto.cols,
            self.dto.n_users,
            self.dto.n_items,
            self.dto.size,
            self.dto.matrix_size,
            self.dto.memory_mb,
            self.dto.cost,
            self.dto.sparsity,
            self.dto.density,
            self.dto.filepath,
            self.dto.id,
        )


# ------------------------------------------------------------------------------------------------ #


@dataclass
class ReadDataset(SQL):
    id: int
    name: str = "dataset"
    sql: str = """SELECT * FROM dataset WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #


@dataclass
class ReadAllDatasets(SQL):
    name: str = "dataset"
    sql: str = """SELECT * FROM dataset;"""
    args: tuple = ()


# ------------------------------------------------------------------------------------------------ #


@dataclass
class DatasetExists(SQL):
    id: int
    name: str = "dataset"
    sql: str = """SELECT EXISTS(SELECT 1 FROM dataset WHERE id = ? LIMIT 1);"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DeleteDataset(SQL):
    id: int
    name: str = "dataset"
    sql: str = """DELETE FROM dataset WHERE id = ?;"""
    args: tuple = ()

    def __post_init__(self) -> None:
        self.args = (self.id,)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Respond:
    row: list

    def as_dto(self) -> None:
        return DatasetDTO(
            id=self.row[0],
            name=self.row[1],
            type=self.row[2],
            description=self.row[3],
            lab=self.row[4],
            stage=self.row[5],
            rows=self.row[6],
            cols=self.row[7],
            n_users=self.row[8],
            n_items=self.row[9],
            size=self.row[10],
            matrix_size=self.row[11],
            memory_mb=self.row[12],
            cost=self.row[13],
            sparsity=self.row[14],
            density=self.row[15],
            filepath=self.row[16],
            created=self.row[17],
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetAdapter(Adapter):
    create: SQL = CreateDatasetTable()
    drop: SQL = DropDatasetTable()
    table_exists: SQL = DatasetTableExists()
    insert: type[SQL] = InsertDataset
    update: type[SQL] = UpdateDataset
    read: type[SQL] = ReadDataset
    read_all: type[SQL] = ReadAllDatasets
    row_exists: type[SQL] = DatasetExists
    delete: type[SQL] = DeleteDataset
    respond: DatasetDTO = Respond
