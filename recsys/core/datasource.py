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
# Modified   : Wednesday February 22nd 2023 11:19:44 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from recsys.core.base import Entity
from recsys.dal.dto import DatasourceDTO


# ------------------------------------------------------------------------------------------------ #
#                                  DATASOURCE ABSTRACT BASE CLASS                                  #
# ------------------------------------------------------------------------------------------------ #
class Datasource(Entity):
    """Datasource base class.

    Args:
        name (str): Name of the data object
        title (str): Title for the data source
        description (str): Describes the contents of the data object

    """

    def __init__(self, name: str, description: str, stage: str) -> None:
        self._name = name
        self._description = description
        self._type = self.__class__.__name__
        self._id = None
        self._filepath = None
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def authors(self) -> str:
        return self._authors

    @authors.setter
    def authors(self, authors: str) -> None:
        self._authors = authors

    @property
    def publisher(self) -> str:
        return self._publisher

    @publisher.setter
    def publisher(self, publisher: str) -> None:
        self._publisher = publisher

    @property
    def published(self) -> str:
        return self._published

    @published.setter
    def published(self, published: str) -> None:
        self._published = published

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    def version(self, version: str) -> None:
        self._version = version

    @property
    def website(self) -> str:
        return self._website

    @website.setter
    def website(self, website) -> None:
        self._website = website

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, url) -> None:
        self._url = url

    @property
    def doi(self) -> str:
        return self._doi

    @doi.setter
    def doi(self, doi) -> None:
        self._doi = doi

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, email) -> None:
        self._email = email

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
