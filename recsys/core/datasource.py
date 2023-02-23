#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/core/datasource.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 04:35:47 pm                                            #
# Modified   : Wednesday February 22nd 2023 06:17:55 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.core.base import DatasourceABC
from recsys.dal.dto import DatasourceDTO


# ------------------------------------------------------------------------------------------------ #
class Datasource(DatasourceABC):
    """Datasource containing movielens ratings data"""

    def as_dto(self) -> DatasourceDTO:
        """Returns a Data Transfer Object representation of the entity."""
        return DatasourceDTO(
            id=self._id,
            name=self._name,
            type=self._type,
            title=self._title,
            description=self._description,
            author=self._author,
            publisher=self._publisher,
            published=self._published,
            version=self._version,
            website=self._website,
            uri=self._uri,
            doi=self._doi,
            email=self._email,
        )
