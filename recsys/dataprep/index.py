#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/index.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 06:41:01 am                                                #
# Modified   : Sunday March 19th 2023 01:19:43 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Prep: Index Module"""
from typing import Union


from pandas import pd
import numpy as np

from recsys import Dataset
from dataprep.operator import Operator


# ------------------------------------------------------------------------------------------------ #
#                            MINIMUM ITEMS PER USER                                                #
# ------------------------------------------------------------------------------------------------ #
class IndexSequencer(Operator):
    """Creates a user and item indices in the sequential space.

    Args:
        useridx (str): The column to contain the new user index.
        itemidx (str: The column to contain the new item index.
        userid (str): Name of the column containing the user id.
        itemid (str): Name of the column containing the item id.

    """

    def __init__(
        self,
        useridx: str = "useridx",
        itemidx: str = "itemidx",
        userid: str = "userId",
        itemid: str = "movieId",
    ) -> None:
        super().__init__()

        self._useridx = useridx
        self._itemidx = itemidx
        self._userid = userid
        self._itemid = itemid

    def __call__(self, data: Union[pd.DataFrame, Dataset]) -> pd.DataFrame:
        """Filters the user interactions by the number of items per user

        Args:
            data (pd.DataFrame) The user rating interaction dataframe.
        """
        if "itemidx" in data.columns:
            msg = "The dataset has already been reindexed."
            self._logger.info(msg)
        else:
            data = self._reindex(data=data, id=self._userid, to=self._useridx)
            return self._reindex(data=data, id=self._itemid, to=self._itemidx)

    def _reindex(self, data: pd.DataFrame, id: str, to: str) -> pd.DataFrame:
        """Creates sequential ids for users and movies."""
        # Get unique user or movie ids.
        features = np.sort(data[id].unique())
        features = pd.DataFrame(data=features, columns=[id])
        features.reset_index(inplace=True)
        features = features.rename(columns={"index": to})
        return data.merge(features, how="left", on=id)
