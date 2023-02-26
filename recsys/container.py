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
# Modified   : Sunday February 26th 2023 05:35:32 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.io.service import IOService
from recsys.persistence.odb import CacheDB, ObjectDB
from recsys.persistence.repo import Repo


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    fio = providers.Factory(IOService)


# ------------------------------------------------------------------------------------------------ #
class PresidioContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    db = providers.Factory(ObjectDB, filepath=config.database)

    cache = providers.Factory(CacheDB, filepath=config.cache, duration=config.duration)

    repo = providers.Factory(Repo, database=db)


# ------------------------------------------------------------------------------------------------ #
class EnricoContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    db = providers.Factory(ObjectDB, filepath=config.database)

    cache = providers.Factory(CacheDB, filepath=config.cache, duration=config.duration)

    repo = providers.Factory(Repo, database=db)


# ------------------------------------------------------------------------------------------------ #
class BackflipContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    db = providers.Factory(ObjectDB, filepath=config.database)

    cache = providers.Factory(CacheDB, filepath=config.cache, duration=config.duration)

    repo = providers.Factory(Repo, database=db)


# ------------------------------------------------------------------------------------------------ #
class DevStudioContainer(containers.DeclarativeContainer):

    presidio = providers.Dependency()

    enrico = providers.Dependency()

    backflip = providers.Dependency()

    config = providers.Configuration()


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    load_dotenv()

    config = providers.Configuration(yaml_files=["config.yml"])

    services = providers.Container(ServicesContainer, config=config)

    presidio = providers.Container(PresidioContainer, config=config.workspaces.presidio)

    enrico = providers.Container(EnricoContainer, config=config.workspaces.enrico)

    backflip = providers.Container(BackflipContainer, config=config.workspaces.backflip)

    devstudio = providers.Container(
        DevStudioContainer,
        presidio=presidio.repo,
        enrico=enrico.repo,
        backflip=backflip.repo,
        config=config,
    )

    repo = providers.Selector(
        config.workspace,
        presidio=providers.Factory(Repo, database=presidio.db),
        enrico=providers.Factory(Repo, database=enrico.db),
        backflip=providers.Factory(Repo, database=backflip.db),
    )

    cache = providers.Selector(
        config.workspace,
        presidio=providers.Factory(
            CacheDB, filepath=config.presidio.cache, duration=config.presidio.duration
        ),
        enrico=providers.Factory(
            CacheDB, filepath=config.enrico.cache, duration=config.enrico.duration
        ),
        backflip=providers.Factory(
            CacheDB, filepath=config.backflip.cache, duration=config.backflip.duration
        ),
    )

    config.override({"workspace": os.getenv("WORKSPACE")})
    instance = cache()
    assert isinstance(instance, CacheDB)
    logging.debug(config)
