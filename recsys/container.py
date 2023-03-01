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
# Modified   : Tuesday February 28th 2023 08:01:37 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from recsys.persistence.io import IOService
from recsys.persistence.odb import ObjectDB
from recsys.persistence.repo import Repo, IDGen
from recsys.persistence.cache import Cache, CacheConfig
from recsys.persistence.datastore import DataStore
from recsys.persistence.workspace import Workspace


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    fio = providers.Factory(IOService)

    iddb = providers.Singleton(ObjectDB, filepath=config.idgen.database)

    idgen = providers.Singleton(IDGen, database=iddb)


# ------------------------------------------------------------------------------------------------ #
class PresidioContainer(containers.DeclarativeContainer):

    config = providers.Configuration()
    idgen = providers.Dependency()

    # Database
    dataset_odb = providers.Factory(ObjectDB, filepath=config.data.dataset)
    experiment_odb = providers.Factory(ObjectDB, filepath=config.data.experiment)
    model_odb = providers.Factory(ObjectDB, filepath=config.data.model)

    # Repositories
    dataset = providers.Factory(Repo, database=dataset_odb, idgen=idgen)
    experiment = providers.Factory(Repo, database=experiment_odb, idgen=idgen)
    model = providers.Factory(Repo, database=model_odb, idgen=idgen)

    # DataStore
    datastore = providers.Factory(
        DataStore,
        workspace=config.name,
        dataset=dataset,
        experiment=experiment,
        model=model,
        datasize=config.data.datasize,
    )

    # Cache Config
    cache_config = providers.Factory(
        CacheConfig,
        workspace=config.name,
        ttl=config.cache.ttl,
        max_size=config.cache.max_size,
        location=config.cache.location,
        enabled=config.cache.enabled,
    )

    cache_db = providers.Factory(ObjectDB, filepath=config.cache.location)

    cache_mgr = providers.Factory(Cache, config=cache_config, storage=cache_db)


# ------------------------------------------------------------------------------------------------ #
class EnricoContainer(containers.DeclarativeContainer):

    config = providers.Configuration()
    idgen = providers.Dependency()

    # Database
    dataset_odb = providers.Factory(ObjectDB, filepath=config.data.dataset)
    experiment_odb = providers.Factory(ObjectDB, filepath=config.data.experiment)
    model_odb = providers.Factory(ObjectDB, filepath=config.data.model)

    # Repositories
    dataset = providers.Factory(Repo, database=dataset_odb, idgen=idgen)
    experiment = providers.Factory(Repo, database=experiment_odb, idgen=idgen)
    model = providers.Factory(Repo, database=model_odb, idgen=idgen)

    # DataStore
    datastore = providers.Factory(
        DataStore,
        workspace=config.name,
        dataset=dataset,
        experiment=experiment,
        model=model,
        datasize=config.data.datasize,
    )

    # Cache Config
    cache_config = providers.Factory(
        CacheConfig,
        workspace=config.name,
        ttl=config.ttl,
        max_size=config.max_size,
        location=config.location,
        enabled=config.enabled,
    )

    cache_db = providers.Factory(ObjectDB, filepath=config.cache.location)

    cache_mgr = providers.Factory(Cache, config=cache_config, storage=cache_db)


# ------------------------------------------------------------------------------------------------ #
class BackflipContainer(containers.DeclarativeContainer):

    config = providers.Configuration()
    idgen = providers.Dependency()

    # Database
    dataset_odb = providers.Factory(ObjectDB, filepath=config.data.dataset)
    experiment_odb = providers.Factory(ObjectDB, filepath=config.data.experiment)
    model_odb = providers.Factory(ObjectDB, filepath=config.data.model)

    # Repositories
    dataset = providers.Factory(Repo, database=dataset_odb, idgen=idgen)
    experiment = providers.Factory(Repo, database=experiment_odb, idgen=idgen)
    model = providers.Factory(Repo, database=model_odb, idgen=idgen)

    # DataStore
    datastore = providers.Factory(
        DataStore,
        workspace=config.name,
        dataset=dataset,
        experiment=experiment,
        model=model,
        datasize=config.data.datasize,
    )

    # Cache Config
    cache_config = providers.Factory(
        CacheConfig,
        workspace=config.name,
        ttl=config.ttl,
        max_size=config.max_size,
        location=config.location,
        enabled=config.enabled,
    )

    cache_db = providers.Factory(ObjectDB, filepath=config.cache.location)

    cache_mgr = providers.Factory(Cache, config=cache_config, storage=cache_db)


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    load_dotenv()

    config = providers.Configuration(yaml_files=["config.yml"])

    services = providers.Container(ServicesContainer, config=config)

    presidio = providers.Container(
        PresidioContainer, config=config.workspaces.presidio, idgen=services.idgen
    )

    enrico = providers.Container(EnricoContainer, config=config.workspaces.enrico)

    backflip = providers.Container(BackflipContainer, config=config.workspaces.backflip)

    workspace = providers.Selector(
        config.workspace,
        presidio=providers.Factory(
            Workspace,
            name=config.workspaces.presidio.name,
            description=config.workspaces.presidio.description,
            cache=presidio.cache_mgr,
            datastore=presidio.datastore,
        ),
        enrico=providers.Factory(
            Workspace,
            name=config.workspaces.enrico.name,
            description=config.workspaces.enrico.description,
            cache=enrico.cache_mgr,
            datastore=enrico.datastore,
        ),
        backflip=providers.Factory(
            Workspace,
            name=config.workspaces.backflip.name,
            description=config.workspaces.backflip.description,
            cache=backflip.cache_mgr,
            datastore=backflip.datastore,
        ),
    )

    config.override({"workspace": os.getenv("WORKSPACE")})
