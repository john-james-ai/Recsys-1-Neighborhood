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
# Modified   : Monday February 20th 2023 11:24:33 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.repo.rating import RatingRepo
from recsys.repo.user import UserRepo
from recsys.repo.task import TaskRepo
from recsys.repo.item import ItemRepo
from recsys.repo.matrix import MatrixRepo
from recsys.dal.dao import DAO
from recsys.dal.fao import FAO
from recsys.data.fio import IOService
from recsys.data.fms import FMS
from recsys.data.rdbms import SQLite


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

    fms = providers.Resource(FMS, io=io)

    db = providers.Singleton(SQLite, config=config.sqlite)


# ------------------------------------------------------------------------------------------------ #
class DALContainer(containers.DeclarativeContainer):

    dbms = providers.Resource()

    fms = providers.Resource()

    dao = providers.Container(DAO, db=dbms)

    fao = providers.Container(FAO, fms=fms)


# ------------------------------------------------------------------------------------------------ #
class RepoContainer(containers.DeclarativeContainer):

    dao = providers.Resource()

    fao = providers.Resource()

    rating = providers.Container(RatingRepo, dao=dao, fao=fao)

    user = providers.Container(UserRepo, dao=dao, fao=fao)

    item = providers.Container(ItemRepo, dao=dao, fao=fao)

    task = providers.Container(TaskRepo, dao=dao, fao=fao)

    metric = providers.Container(MatrixRepo, dao=dao, fao=fao)


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yml"])

    logging = providers.Container(LoggingContainer, config=config)

    data = providers.Container(DataContainer, config=config)

    dal = providers.Container(DALContainer, dbms=data.db, fms=data.fms)

    repo = providers.Container(RepoContainer, dao=dal.dao, fao=dal.fao)
