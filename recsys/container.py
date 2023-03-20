#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/container.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 09:25:49 am                                                 #
# Modified   : Monday March 20th 2023 04:07:27 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.services.io import IOService
from recsys.persistence.database import Database

from recsys.asset.repo import AssetRepo
from recsys.asset.centre import AssetCentre


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    database = providers.Singleton(
        Database,
        directory=config.database.directory,
        filename=config.database.filename,
    )


# ------------------------------------------------------------------------------------------------ #
class RepoContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    database = providers.Dependency()

    datasource = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.datasource.directory,
        tablename=config.datasource.tablename,
        io=IOService,
    )

    dataset = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.dataset.directory,
        tablename=config.dataset.tablename,
        io=IOService,
    )

    operator = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.operator.directory,
        tablename=config.operator.tablename,
        io=IOService,
    )

    algorithm = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.algorithm.directory,
        tablename=config.algorithm.tablename,
        io=IOService,
    )

    model = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.model.directory,
        tablename=config.model.tablename,
        io=IOService,
    )

    pipeline = providers.Singleton(
        AssetRepo,
        database=database,
        directory=config.pipeline.directory,
        tablename=config.pipeline.tablename,
        io=IOService,
    )

    centre = providers.Singleton(
        AssetCentre,
        datasource=datasource,
        dataset=dataset,
        operator=operator,
        algorithm=algorithm,
        model=model,
        pipeline=pipeline,
    )


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["tests/config.yml"])

    services = providers.Container(ServicesContainer, config=config)

    assets = providers.Container(RepoContainer, config=config.assets, database=services.database)
