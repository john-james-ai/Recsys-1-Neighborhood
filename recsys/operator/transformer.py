#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/transformer.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:50:55 pm                                            #
# Modified   : Friday February 24th 2023 11:13:21 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Transformer Module"""
import pandas as pd

from recsys.operator.base import Operator
from recsys.workflow.event import event_log


class RatingCenterer(Operator):
    """Centers ratings by average user and item rating

    Args:
        source (str): The source filepath
        destination (str): Destination filepath
        uservar (str): The name of the column containing the user identifier.
        itemvar (str): The name of the column containing the item identifier.
        user_centered_rating_var (str): The name of the column containing the user centered rating
        item_centered_rating_var (str): The name of the column containing the item centered rating
        force (bool): Wether to force execution
    """

    def __init__(
        self,
        source: str,
        destination: str,
        uservar: str,
        itemvar: str,
        rating_var: str,
        user_centered_rating_var: str,
        item_centered_rating_var: str,
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._uservar = uservar
        self._itemvar = itemvar
        self._rating_var = rating_var
        self._user_centered_rating_var = user_centered_rating_var
        self._item_centered_rating_var = item_centered_rating_var

    @event_log
    def run(self) -> None:

        data = self._fio.read(filepath=self._source)
        data = self._center_by_user_rating(data)
        data = self._center_by_item_rating(data)
        self._fio.write(filepath=self._destination, data=data)

    def _center_by_user_rating(self, data: pd.DataFrame) -> pd.DataFrame:
        """Centers rating by average user rating"""
        # Obtain average user ratings
        rbar = data.groupby(self._uservar)[self._rating_var].mean().reset_index()
        rbar.columns = [self._uservar, "rbar"]
        # Merge with ratings dataset
        data = data.merge(rbar, on=self._uservar, how="left")
        # Compute centered rating and drop the average rating column
        data[self._user_centered_rating_var] = data[self._rating_var] - data["rbar"]
        data = data.drop(columns=["rbar"])

        return data

    def _center_by_item_rating(self, data: pd.DataFrame) -> pd.DataFrame:
        """Centers rating by average item rating"""
        # Obtain average item ratings
        rbar = data.groupby(self._itemvar)[self._rating_var].mean().reset_index()
        rbar.columns = [self._itemvar, "rbar"]
        # Merge with ratings dataset
        data = data.merge(rbar, on=self._itemvar, how="left")
        # Compute centered rating and drop the average rating column
        data[self._item_centered_rating_var] = data[self._rating_var] - data["rbar"]
        data = data.drop(columns=["rbar"])

        return data
