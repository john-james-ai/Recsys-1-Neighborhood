#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/cooccurrence.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Recsys-1-Neighborhood                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 2nd 2023 10:01:36 pm                                                 #
# Modified   : Sunday March 5th 2023 07:38:09 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Corating Module"""
from abc import abstractmethod
from functools import cache

import numpy as np
from tqdm import tqdm
import pandas as pd
from recsys.services.combinations import cartesian_product

from recsys.operator.base import Operator, Artifact


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
    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
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
        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    @cache
    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:

        cooccurrence = {}

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:
                # Obtain the unique items in the ratings dataset
                for item, ratings in tqdm(data.groupby(self._itemvar)):
                    # For each item, get the users whom have rated the item.
                    users = ratings[self._uservar].values
                    if len(users) > 1:
                        # Create 2 arrays for the cartesian product function
                        a = np.array(users)
                        b = np.array(users)
                        # Compute the cartesian product of the users
                        user_pairs = cartesian_product(a, b)
                        # Transpose
                        user_pairs = user_pairs.T
                        # Convert list of arrays to tuples
                        user_pairs = list(zip(user_pairs[:, 0], user_pairs[:, 1]))
                        # iterate over unique tuple combinations of users
                        for user_pair in user_pairs:
                            # Sort pairs to and avoid duplicates
                            user_pair = tuple(sorted(user_pair))
                            # Drop matching pairs and eliminate duplicates
                            if user_pair[0] != user_pair[1]:

                                a = ratings[ratings["userId"] == user_pair[0]][
                                    ["userId", "movieId"]
                                ]
                                b = ratings[ratings["userId"] == user_pair[1]][
                                    ["userId", "movieId"]
                                ]
                                common_items = a[["userId", "movieId"]].merge(
                                    b[["userId", "movieId"]], on="movieId", how="inner"
                                )
                                if cooccurrence.get(user_pair, None) is None:
                                    cooccurrence[user_pair] = list(common_items["movieId"].values)
                                else:
                                    cooccurrence[user_pair].extend(
                                        list(common_items["movieId"].values)
                                    )
            except Exception as e:  # pragma : no cover
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
        self._artifact = {"item_cooccurrence_index": self._destination}

    @cache
    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:

        cooccurrence = {}

        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:
                # Obtain the unique users in the ratings dataset
                for user, ratings in tqdm(data.groupby(self._uservar)):
                    # For each user, get the items rated.
                    items = ratings[self._itemvar].values
                    for item in items:
                        # Check other raters
                        users = data[data[self._itemvar] == item][self._uservar].values
                        # Debugging code
                        self._logger.debug(f"\n\nUser {user} has {len(users)-1} neighbors")
                        # Find cartesian product of users
                        a = np.array(users)
                        b = np.array(users)
                        # Compute the cartesian product of the items
                        user_pairs = cartesian_product(a, b)
                        # Debugging code
                        self._logger.debug(
                            f"\nThis in turn equates to {len(user_pairs)} combinations."
                        )
                        # Transpose
                        user_pairs = user_pairs.T
                        # Convert list of arrays to tuples
                        user_pairs = list(zip(user_pairs[:, 0], user_pairs[:, 1]))
                        # iterate over unique tuple combinations of users
                        for user_pair in user_pairs:
                            # Drop matching pairs and eliminate duplicates
                            if user_pair[0] != user_pair[1]:
                                user_pair = tuple(sorted(user_pair))
                                # Add the item as cooccurring with the two users
                                if cooccurrence.get(user_pair, None) is None:
                                    cooccurrence[user_pair] = [item]
                                else:
                                    cooccurrence[user_pair].append(item)

            except Exception as e:  # pragma : no cover
                self._logger.error(e)
                raise

            self._put_data(filepath=self._destination, data=cooccurrence)

            return cooccurrence
