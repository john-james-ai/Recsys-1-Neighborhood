#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_container/test_devstudio.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 12:17:58 pm                                               #
# Modified   : Tuesday February 28th 2023 08:01:37 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from recsys.persistence.cache import Cache
from recsys.persistence.datastore import DataStore
from recsys.persistence.workspace import Workspace
from recsys.persistence.repo import Repo

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.box
class TestContainer:  # pragma: no cover
    # ============================================================================================ #
    def test_workspace(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Presidio
        assert isinstance(container.workspace.presidio(), Workspace)
        assert isinstance(container.workspace.presidio().name, str)
        assert isinstance(container.workspace.presidio().description, str)
        assert isinstance(container.workspace.presidio().cache, Cache)
        assert isinstance(container.workspace.presidio().datastore, DataStore)
        assert isinstance(container.workspace.presidio().datastore.dataset, Repo)

        # Enrico
        assert isinstance(container.workspace.enrico(), Workspace)
        assert isinstance(container.workspace.enrico().name, str)
        assert isinstance(container.workspace.enrico().description, str)
        assert isinstance(container.workspace.enrico().cache, Cache)
        assert isinstance(container.workspace.enrico().datastore, DataStore)
        assert isinstance(container.workspace.enrico().datastore.dataset, Repo)

        # Backflip
        assert isinstance(container.workspace.backflip(), Workspace)
        assert isinstance(container.workspace.backflip().name, str)
        assert isinstance(container.workspace.backflip().description, str)
        assert isinstance(container.workspace.backflip().cache, Cache)
        assert isinstance(container.workspace.backflip().datastore, DataStore)
        assert isinstance(container.workspace.backflip().datastore.dataset, Repo)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
