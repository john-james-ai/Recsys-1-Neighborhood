#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/repo/base.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:03:56 pm                                            #
# Modified   : Wednesday February 22nd 2023 11:30:06 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Any
from recsys.core.base import Entity
from recsys.repo.uow import UnitOfWork


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Definition of the Repository interface."""

    @property
    def uow(self) -> UnitOfWork:
        """Returns the unit of work instance."""

    @uow.setter
    def uow(self, uow: UnitOfWork) -> UnitOfWork:
        """Unit of Work"""

    @abstractmethod
    def get(self, id) -> Any:
        """Gets an item by id."""

    @abstractmethod
    def add(self, entity: Entity) -> None:
        """Add an entity to the repository."""

    @abstractmethod
    def remove(self, id: int) -> int:
        """Remove an entity instance from the repository."""

    @abstractmethod
    def print(self, i: int = None) -> Entity:
        """Print one or more entities.."""
