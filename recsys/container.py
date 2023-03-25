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
# Modified   : Monday March 20th 2023 10:43:38 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.services.io import IOService
from recsys.persistence.database import Database

from recsys.asset.centre import AssetCentre


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    database = providers.Singleton(Database, filepath=config.assets.database)


# ------------------------------------------------------------------------------------------------ #
class AssetCentreContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    database = providers.Dependency()

    centre = providers.Singleton(
        AssetCentre,
        database=database,
        directory=config.directory,
        tablename=config.tablename,
        io=IOService,
    )


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["tests/config.yml"])

    services = providers.Container(ServicesContainer, config=config)

    asset = providers.Container(
        AssetCentreContainer, config=config.assets, database=services.database
    )
