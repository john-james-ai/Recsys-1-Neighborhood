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
# Modified   : Sunday February 19th 2023 05:23:08 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.data.rdbms import SQLite
from recsys.data.fio import IOService
from recsys.data.fms import FileService, ModelService
from recsys.dal.dao import DatasetDAO, ModelDAO
from recsys.dal.fao import FAO
from recsys.dal.mao import MAO
from recsys.dal.dataset import DatasetRepo
from recsys.dal.model import ModelRepo
from recsys.dal.sql import DatasetDML


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

    io = providers.Resource(IOService)

    db = providers.Resource(SQLite, config=config.sqlite)

    fms = providers.Resource(FileService, config=config.files, io=io)

    mms = providers.Resource(ModelService, config=config.files, io=io)


# ------------------------------------------------------------------------------------------------ #
class DALContainer(containers.DeclarativeContainer):

    db = providers.Dependency()
    fms = providers.Dependency()
    mms = providers.Dependency()

    dataset_dao = providers.Resource(DatasetDAO, db=db, dml=DatasetDML)

    fao = providers.Resource(FAO, fms=fms)

    mao = providers.Resource(MAO, mms=mms)


# ------------------------------------------------------------------------------------------------ #
class RepoContainer(containers.DeclarativeContainer):

    dao = providers.Dependency()
    fao = providers.Dependency()
    mao = providers.Dependency()

    dataset = providers.Resource(DatasetRepo, dao=dao, fao=fao)

    model = providers.Resource(ModelRepo, dao=dao, mao=mao)


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yml"])

    logging = providers.Container(LoggingContainer, config=config)

    data = providers.Container(DataContainer, config=config)

    dal = providers.Container(DALContainer, db=data.db, fms=data.fms, mms=data.mms)

    repo = providers.Container(RepoContainer, dao=dal.dao, fao=dal.fao, mao=dal.mao)
