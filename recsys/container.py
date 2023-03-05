#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/container.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 4th 2023 09:25:49 am                                                 #
# Modified   : Saturday March 4th 2023 07:10:53 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers


# ------------------------------------------------------------------------------------------------ #
class ServicesContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config/logging.yml", "config/etl.pml"])

    services = providers.Container(ServicesContainer, config=config)
