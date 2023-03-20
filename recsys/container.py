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
# Modified   : Sunday March 19th 2023 08:34:49 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.services.io import IOService
from recsys.persistence.database import Database
from recsys.persistence.centre import AssetCentre


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    dbconn = providers.Singleton(Database.connection)


# ------------------------------------------------------------------------------------------------ #
class AssetCenterContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    cxn = providers.Dependency()

    datasource = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.datasource.location,
        tablename=config.datasource.tablename,
        io=IOService,
    )

    dataset = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.dataset.location,
        tablename=config.dataset.tablename,
        io=IOService,
    )

    operator = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.operator.location,
        tablename=config.operator.tablename,
        io=IOService,
    )

    algorithm = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.algorithm.location,
        tablename=config.algorithm.tablename,
        io=IOService,
    )

    model = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.model.location,
        tablename=config.model.tablename,
        io=IOService,
    )

    pipeline = providers.Singleton(
        AssetCentre,
        cxn=cxn,
        location=config.pipeline.location,
        tablename=config.pipeline.tablename,
        io=IOService,
    )


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["tests/config.yml"])

    services = providers.Container(ServicesContainer, config=config)

    assets = providers.Container(AssetCenterContainer, config=config.assets, cxn=services.dbconn)
