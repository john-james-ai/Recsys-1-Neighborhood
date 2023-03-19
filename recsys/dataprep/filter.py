#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/filter.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 06:39:05 am                                                #
# Modified   : Saturday March 18th 2023 09:00:11 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Prep: Filter Module"""
from typing import Union

from tqdm import tqdm
from pandas import pd
import mlflow

from recsys import Dataset

from dataprep.operator import Operator


# ------------------------------------------------------------------------------------------------ #
#                            MINIMUM ITEMS PER USER                                                #
# ------------------------------------------------------------------------------------------------ #
class MinItemsPerUserFilter(Operator):
    """Filters users with reviews below the stated threshold

    Args:
        min_items_per_user (int): The minimum number of items per user required. Default = 4
        drop_duplicates (bool): Whether to drop duplicate interactions beetween a user
            and an item. Default = True
        userid (str): Name of the column containing the user id.
        itemid (str): Name of the column containing the item id.

    """

    def __init__(
        self,
        min_items_per_user: int = 4,
        drop_duplicates: bool = True,
        userid: str = "userId",
        itemid: str = "movieId",
    ) -> None:
        super().__init__()

        self._min_items_per_user = min_items_per_user
        self._drop_duplicates = drop_duplicates
        self._userid = userid
        self._itemid = itemid
        mlflow.log_param(key="min_items_per_user", value=self._min_items_per_user)
        mlflow.log_param(key="drop_duplicates", value=self._drop_duplicates)

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Filters the user interactions by the number of items per user

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """
        userids = (
            data.drop_duplicates([self._userid, self._itemid])[self._userid]
            if self._drop_duplicates
            else data[self._userid]
        )
        items_per_user = userids.value_counts()
        users_to_keep = list(items_per_user[items_per_user >= self._min_items_per_user])

        return data[data[self._userid].isin(users_to_keep)].copy()


# ------------------------------------------------------------------------------------------------ #
#                            MINIMUM USERS PER ITEM                                                #
# ------------------------------------------------------------------------------------------------ #
class MinUsersPerItemFilter(Operator):
    """Filters items based upon the number of user interactions.

    Args:
        min_users_per_item (int): The minimum number of items per user required. Default = 4
        drop_duplicates (bool): Whether to drop duplicate interactions beetween a user
            and an item. Default = True
        userid (str): Name of the column containing the user id.
        itemid (str): Name of the column containing the item id.

    """

    def __init__(
        self,
        min_users_per_item: int = 4,
        drop_duplicates: bool = True,
        userid: str = "userId",
        itemid: str = "movieId",
    ) -> None:

        self._min_users_per_item = min_users_per_item
        self._drop_duplicates = drop_duplicates
        self._userid = userid
        self._itemid = itemid
        mlflow.log_param(key="min_users_per_item", value=self._min_items_per_user)
        mlflow.log_param(key="drop_duplicates", value=self._drop_duplicates)

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Filters the items with interactions below a threshold

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """
        itemids = (
            data.drop_duplicates([self._userid, self._itemid])[self._itemid]
            if self._drop_duplicates
            else data[self._itemid]
        )
        users_per_item = itemids.value_counts()
        items_to_keep = list(users_per_item[users_per_item >= self._min_users_per_item])

        return data[data[self._userid].isin(items_to_keep)].copy()


# ------------------------------------------------------------------------------------------------ #
#                            MAXIMUM ITEMS PER USER                                                #
# ------------------------------------------------------------------------------------------------ #
class MaxItemsPerUserFilter(Operator):
    """Filters users based upon a maximum number of interactions.

    The first 'max_items_per_user' interactions are obtained for each user. For each additional
    user interaction, a randomly chosen interaction from the user's history is discarded,
    and replaced by the new interaction. This process repeats until all interactions for
    each user are processed in this way. Adapted from [1_].

    Args:
        max_items_per_user (int): The maximum number of items allowed per user. Default = 1000
        drop_duplicates (bool): Whether to drop duplicate interactions beetween a user
            and an item. Default = True
        userid (str): Name of the column containing the user id.
        itemid (str): Name of the column containing the item id.
        timestamp (timestamp): Timestamp of the interaction

    Reference:
    .. [1] S. Schelter, U. Celebi, and T. Dunning, “Efficient Incremental Cooccurrence
    Analysis for Item-Based Collaborative Filtering,” in Proceedings of the 31st International
    Conference on Scientific and Statistical Database Management, Santa Cruz CA
    USA, Jul. 2019, pp. 61–72. doi: 10.1145/3335783.3335784.


    """

    def __init__(
        self,
        max_items_per_user: int = 1000,
        drop_duplicates: bool = True,
        userid: str = "userId",
        itemid: str = "movieId",
        timestamp: str = "timestamp",
    ) -> None:

        self._max_items_per_user = max_items_per_user
        self._drop_duplicates = drop_duplicates
        self._userid = userid
        self._itemid = itemid
        self._timestamp = timestamp
        self._interactions_cut = 0
        mlflow.log_param(key="max_items_per_user", value=self._max_items_per_user)
        mlflow.log_param(key="drop_duplicates", value=self._drop_duplicates)

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Filters the user interactions above a threshold

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """

        # Sample the first user interactions, up to the threshold
        data = self._sample_interactions(data=data)

        data.drop(columns=["interactions"], inplace=True)

        self._logger.debug(f"\nInteractions cut: {self._interactions_cut}")

        return data

    def _sample_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Obtain the first transactions up to the threshold."""

        # Groupby user, sort by timestamp ascending within groups, and count the interactions.
        data["interactions"] = (
            data.sort_values(by=self._timestamp, ascending=True)
            .groupby(by=self._userid, group_keys=False)
            .cumcount()
            + 1
        )

        # Retain the first interactions up to the  threshold
        data = data[data["interactions"] <= self._max_items_per_user]
        pending = data[data["interactions"] > self._max_items_per_user]

        if pending.shape[0] > 0:
            data = self._handle_pending(data=data, pending=pending)
        return data

    def _handle_pending(self, data: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
        """Existing interactions are randomly withdrawn to accommodate new interactions."""

        # Repeatedly remove a single interaction at random and replace with new interaction.
        groups = pending.sort_values(by=[self._timestamp], ascending=True).groupby(
            by=self._userid, group_keys=False
        )
        for name, ratings in tqdm(groups):
            for _, row in ratings.iterrows():
                row = row.to_frame()
                eviction = data[data[self._userid] == name].sample(n=1, replace=False, axis=0)
                self._interactions_cut += 1
                data = data.drop(eviction.index.values)
                data = pd.concat([data, row.T], axis=0)
        return data


# ------------------------------------------------------------------------------------------------ #
#                            MAXIMUM USERS PER ITEM                                                #
# ------------------------------------------------------------------------------------------------ #
class MaxUsersPerItemFilter(Operator):
    """Filters items based upon a maximum number of interactions.

    The first 'max_users_per_item' interactions are obtained for each item. For each additional
    item interaction, a randomly chosen interaction from the item's history is discarded,
    and replaced by the new interaction. This process repeats until all interactions for
    each item are processed in this way. Adapted from [1_].

    Args:
        max_users_per_item (int): The maximum number of interactions allowed per item. Default = 1000
        drop_duplicates (bool): Whether to drop duplicate interactions beetween a user
            and an item. Default = True
        userid (str): Name of the column containing the user id.
        itemid (str): Name of the column containing the item id.
        timestamp (timestamp): Timestamp of the interaction

    Reference:
    .. [1] S. Schelter, U. Celebi, and T. Dunning, “Efficient Incremental Cooccurrence
    Analysis for Item-Based Collaborative Filtering,” in Proceedings of the 31st International
    Conference on Scientific and Statistical Database Management, Santa Cruz CA
    USA, Jul. 2019, pp. 61–72. doi: 10.1145/3335783.3335784.


    """

    def __init__(
        self,
        max_users_per_item: int = 1000,
        drop_duplicates: bool = True,
        userid: str = "userId",
        itemid: str = "movieId",
        timestamp: str = "timestamp",
    ) -> None:

        self._max_users_per_item = max_users_per_item
        self._drop_duplicates = drop_duplicates
        self._userid = userid
        self._itemid = itemid
        self._timestamp = timestamp
        self._interactions_cut = 0
        mlflow.log_param(key="max_users_per_item", value=self._max_users_per_item)
        mlflow.log_param(key="drop_duplicates", value=self._drop_duplicates)

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Filters the items with interactions above a threshold

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """

        # Sample the first user interactions, up to the threshold
        data = self._sample_interactions(data=data)

        data.drop(columns=["interactions"], inplace=True)

        self._logger.debug(f"\nInteractions cut: {self._interactions_cut}")

        return data

    def _sample_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Obtain the first transactions up to the respective threshold."""

        # Groupby user, sort by timestamp ascending within groups, and count the interactions.
        data["interactions"] = (
            data.sort_values(by=self._timestamp, ascending=True)
            .groupby(by=self._itemid, group_keys=False)
            .cumcount()
            + 1
        )

        # Retain the first interactions up to the  threshold
        data = data[data["interactions"] <= self._max_users_per_item]
        pending = data[data["interactions"] > self._max_users_per_item]

        if pending.shape[0] > 0:
            data = self._handle_pending(data=data, pending=pending)
        return data

    def _handle_pending(self, data: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
        """Existing interactions are randomly withdrawn to accommodate new interactions."""

        # Repeatedly remove a single interaction at random and replace with new interaction.
        groups = pending.sort_values(by=[self._timestamp], ascending=True).groupby(
            by=self._itemid, group_keys=False
        )
        for name, ratings in tqdm(groups):
            for _, row in ratings.iterrows():
                row = row.to_frame()
                eviction = data[data[self._itemid] == name].sample(n=1, replace=False, axis=0)
                self._interactions_cut += 1
                data = data.drop(eviction.index.values)
                data = pd.concat([data, row.T], axis=0)
        return data
