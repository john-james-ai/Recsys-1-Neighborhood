#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/transformation.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:50:55 pm                                            #
# Modified   : Saturday February 25th 2023 09:19:56 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Transformer Module"""
import pandas as pd

from recsys import operator
from recsys.workflow.event import event_log


# ------------------------------------------------------------------------------------------------ #
class MeanCenter(operator.TransformationOperator):
    """Centers ratings by average user and item rating

    Args:
        by (str): The aggregate average rating by which the ratings will be centered. Valid
            values are ['userId','movieId']. Default is 'userId'
    Returns: DataFrame including the centered rating. The column name is automatically set
        to rating_ctr_<by>.
    """

    def __init__(
        self,
        by: str = "userId",
    ) -> None:
        super().__init__()
        self._by = by
        self._column = "rating_ctr_" + by
        self._uservar = uservar
        self._itemvar = itemvar
        self._rating_var = rating_var
        self._user_centered_rating_var = user_centered_rating_var
        self._item_centered_rating_var = item_centered_rating_var

    @event_log
    def run(self, data: pd.DataFrame = None) -> None:

        data = self._fio.read(filepath=self._source)
        data = self._center_by_user_rating(data)
        data = self._center_by_item_rating(data)
        self._fio.write(filepath=self._destination, data=data)

    def _center_by_user_rating(self, data: pd.DataFrame) -> pd.DataFrame:
        """Centers rating by average user rating"""
        try:
            # Obtain average user ratings
            rbar = data.groupby(self._uservar)[self._rating_var].mean().reset_index()
            rbar.columns = [self._uservar, "rbar"]
            # Merge with ratings dataset
            data = data.merge(rbar, on=self._uservar, how="left")
            # Compute centered rating and drop the average rating column
            data[self._user_centered_rating_var] = data[self._rating_var] - data["rbar"]
            data = data.drop(columns=["rbar"])
            return data
        except KeyError as e:
            msg = "User, item, or rating variable is incorrect."
            self._logger.error(msg)
            raise ValueError(e)

    def _center_by_item_rating(self, data: pd.DataFrame) -> pd.DataFrame:
        """Centers rating by average item rating"""
        try:
            # Obtain average item ratings
            rbar = data.groupby(self._itemvar)[self._rating_var].mean().reset_index()
            rbar.columns = [self._itemvar, "rbar"]
            # Merge with ratings dataset
            data = data.merge(rbar, on=self._itemvar, how="left")
            # Compute centered rating and drop the average rating column
            data[self._item_centered_rating_var] = data[self._rating_var] - data["rbar"]
            data = data.drop(columns=["rbar"])
            return data

        except KeyError as e:
            msg = "User, item, or rating variable is incorrect."
            self._logger.error(msg)
            raise ValueError(e)
