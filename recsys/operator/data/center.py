#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/center.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:50:55 pm                                            #
# Modified   : Thursday March 2nd 2023 11:25:48 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Center Module"""
import pandas as pd

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
class MeanCenter(Operator):
    """Centers ratings by average user and item rating

    Args:
        source (str): The URL to the zip file resource
        destination (str): A filename into which the zip file will be stored.
        by (str): The aggregate average rating by which the ratings will be centered. Valid
            values are ['userId','movieId']. Default is 'userId'
        force (bool): Whether to force execution.

    """

    def __init__(
        self,
        by: str = "userId",
        column: str = "rating_ctr_user",
        rating_var: str = "rating",
        source: str = None,
        destination: str = None,
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._by = by
        self._column = column
        self._rating_var = rating_var

    def execute(self, data: pd.DataFrame = None) -> None:

        if not self._skip(endpoint=self._destination):
            # Get the data. Priority goes to data passed into the method.
            data = data or self._get_data(filepath=self._soource)
            try:
                # Obtain average user ratings
                rbar = data.groupby(self._by)[self._rating_var].mean().reset_index()
                rbar.columns = [self._by, "rbar"]

                # Merge with ratings dataset
                data = data.merge(rbar, on=self._by, how="left")

                # Compute centered rating and drop the average rating column
                data[self._column] = data[self._rating_var] - data["rbar"]
                data = data.drop(columns=["rbar"])

                # Save data if destination is provided.
                self._put_data(filepath=self._destination, data=data)

                return data
            except KeyError as e:
                msg = "User, item, or rating variable is incorrect."
                self._logger.error(msg)
                raise ValueError(e)
