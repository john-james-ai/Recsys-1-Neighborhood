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
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 09:10:21 am                                                #
# Modified   : Sunday January 29th 2023 09:26:25 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging.config  # pragma: no cover

from dependency_injector import containers, providers


# ------------------------------------------------------------------------------------------------ #
class LoggingContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
class Recsys(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yml"])

    logging = providers.Container(LoggingContainer, config=config)
