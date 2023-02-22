#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/container.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 09:10:21 am                                                #
# Modified   : Wednesday February 22nd 2023 02:26:24 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.dal.dao import DAO
from recsys.dal.dba import DBA
from recsys.persistence.fio import IOService
from recsys.adapter.dataset import DatasetAdapter
from recsys.persistence.rdbms import SQLite


# ------------------------------------------------------------------------------------------------ #
class LoggingContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
class DataContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    fio = providers.Singleton(IOService)

    db = providers.Singleton(SQLite, config=config.sqlite)

    dao = providers.Resource(DAO, adapter=DatasetAdapter, database=db)


# ------------------------------------------------------------------------------------------------ #
class DBAContainer(containers.DeclarativeContainer):

    db = providers.Configuration()

    dataset = providers.Resource(DBA, adapter=DatasetAdapter, database=db)


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yml"])

    logging = providers.Container(LoggingContainer, config=config)

    data = providers.Container(DataContainer, config=config.database)

    dba = providers.Container(DBAContainer, db=data.db)
