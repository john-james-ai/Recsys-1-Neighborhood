#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/co-occurrence.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 2nd 2023 10:01:36 pm                                                 #
# Modified   : Friday March 3rd 2023 12:47:57 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Corating Module"""
from abc import abstractmethod
from itertools import combinations
from functools import cache
import pandas as pd

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
class CooccurrenceIndex(Operator):
    """Base class for cooccurrence indices.

    Args:
        source (str): The URL to the zip file resource
        destination (str): A filename into which the zip file will be stored.
        uservar (str): Column containing user id
        itemvar (str): Column containing item id
        force (bool): Whether to force execution.

    """

    def __init__(
        self,
        source: str = None,
        destination: str = None,
        uservar: str = "userId",
        itemvar: str = "movieId",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._uservar = uservar
        self._itemvar = itemvar

    @abstractmethod
    def execute(self, data: pd.DataFrame = None) -> None:
        """Creates the cooccurrence index"""


# ------------------------------------------------------------------------------------------------ #
class UserCooccurrenceIndex(CooccurrenceIndex):
    """User coocurrence index. Identifies pairs of users who have rated a particular item.

    Args:
        source (str): The URL to the zip file resource
        destination (str): A filename into which the zip file will be stored.
        uservar (str): Column containing user id
        itemvar (str): Column containing item id
        force (bool): Whether to force execution.

    """

    def __init__(
        self,
        source: str = None,
        destination: str = None,
        uservar: str = "userId",
        itemvar: str = "movieId",
        force: bool = False,
    ) -> None:
        super().__init__(
            source=source, destination=destination, uservar=uservar, itemvar=itemvar, force=force
        )

    @cache
    def execute(self, data: pd.DataFrame = None) -> None:

        cooccurrence = pd.DataFrame()

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:
                # Obtain the unique items in the ratings dataset
                for item, ratings in data.groupby(self._itemvar):
                    # For each item, get the users who have rated the item
                    users = ratings[self._uservar].values
                    # Create unique tuple combinations of users
                    user_pairs = combinations(users, 2)
                    # Format the user pairs and items into a dataframe
                    d = {self._uservar: user_pairs, self._itemvar: item}
                    df = pd.DataFrame(data=d)
                    cooccurrence = pd.concat([cooccurrence, df], axis=0)
            except Exception as e:
                self._logger.error(e)
                raise

            self._put_data(filepath=self._destination, data=cooccurrence)

            return cooccurrence


# ------------------------------------------------------------------------------------------------ #
class ItemCooccurrenceIndex(CooccurrenceIndex):
    """User coocurrence index. Identifies pairs of users who have rated a particular item.

    Args:
        source (str): The URL to the zip file resource
        destination (str): A filename into which the zip file will be stored.
        uservar (str): Column containing user id
        itemvar (str): Column containing item id
        force (bool): Whether to force execution.

    """

    def __init__(
        self,
        source: str = None,
        destination: str = None,
        uservar: str = "userId",
        itemvar: str = "movieId",
        force: bool = False,
    ) -> None:
        super().__init__(
            source=source, destination=destination, uservar=uservar, itemvar=itemvar, force=force
        )

    @cache
    def execute(self, data: pd.DataFrame = None) -> None:

        cooccurrence = pd.DataFrame()

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:
                # Obtain the unique users in the ratings dataset
                for user, ratings in data.groupby(self._uservar):
                    # For each user, get the items which the user has rated.
                    items = ratings[self._itemvar].values
                    # Create unique tuple combinations of items
                    item_pairs = combinations(items, 2)
                    # Format the item pairs and users into a dataframe
                    d = {self._itemvar: item_pairs, self._uservar: user}
                    df = pd.DataFrame(data=d)
                    cooccurrence = pd.concat([cooccurrence, df], axis=0)
            except Exception as e:
                self._logger.error(e)
                raise

            self._put_data(filepath=self._destination, data=cooccurrence)

            return cooccurrence
