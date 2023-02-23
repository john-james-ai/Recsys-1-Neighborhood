#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/workflow.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:54:34 pm                                            #
# Modified   : Wednesday February 22nd 2023 06:21:04 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

# ------------------------------------------------------------------------------------------------ #
class Pipeline(ABC):
    """Base class for Pipeline objects.

    Args:
        name (str): name of the pipeline
        description (str): Description of what the pipeline does.
        io (IOService): Input/output service

    """

    @inject
    def __init__(
        self, name: str, description: str, engine: engine = Provide[Recsys.data.db]
    ) -> None:
        self._name = name
        self._description = description
        self._iengine
        self._tasks = {}

    @abstractmethod
    def add_task(self, operator: Operator) -> None:
        """Add a task performed by a parameterized operator to the pipeline"""

    @logger
    @abstractmethod
    def run(self) -> None:
        """Execute the pipeline"""
