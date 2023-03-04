#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/sampling.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:51:25 pm                                            #
# Modified   : Friday March 3rd 2023 02:48:13 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sampling Operator Module"""
from functools import cache

from tqdm import tqdm
import pandas as pd

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
class UserRandomSampling(Operator):
    """Samples users at random until the target sample size is obtained.

    Data may be sourced from file or passed as a parameter into the run method. Priority
    goes to the parameter passed during runtime into the run method.

    Args:
        frac (str): The fraction of the original dataset to retain
        uservar (str): The name of the column containing the user identifier.
        itemvar (str): The name of the column containing the item identifier.
        random_state (int): Seed for pseudo random sampling
        source (str): Source file path. Optional
        destination (str): The output filepath. Optional
        force (bool): Whether to force execution if the endpoint already exists.
    """

    def __init__(
        self,
        frac: float,
        uservar: str = "userId",
        itemvar: str = "movieId",
        source: str = None,
        destination: str = None,
        random_state: int = None,
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._frac = frac
        self._uservar = uservar
        self._itemvar = itemvar
        self._random_state = random_state

    @cache
    def execute(self, data: pd.DataFrame = None) -> None:
        """Samples the data"""
        if not self._skip(endpoint=self._destination):
            # Get the data. Priority goes to data passed into the method.
            data = data or self._get_data(filepath=self._source)
            # Compute the number of samples to return
            size = int(data.shape[0] * self._frac)
            # Obtain value counts
            counts = data[self._uservar].value_counts(sort=False).to_frame().reset_index()
            counts.columns = [self._uservar, "count"]
            # Shuffle and add cumulative sum of counts
            counts = counts.sample(frac=1, random_state=self._random_state)
            counts["cumsum"] = counts["count"].cumsum()
            # Select users while cumulative sum is <= size
            users = counts[counts["cumsum"] <= size][self._uservar].values
            # Subset the dataframe to include the selected users
            sample = data[data[self._uservar].isin(users)]
            # Save data if destination is provided.
            self._put_data(filepath=self._destination, data=sample)
            # eh viola!
            return sample


# ------------------------------------------------------------------------------------------------ #
class UserStratifiedRandomSampling(Operator):
    """Takes samples from each user in proportion to their interaction.

    Data may be sourced from file or passed as a parameter into the run method. Priority
    goes to the parameter passed during runtime into the run method.

    Args:
        frac (str): The fraction of the original dataset to retain
        uservar (str): The name of the column containing the user identifier.
        itemvar (str): The name of the column containing the item identifier.
        random_state (int): Seed for pseudo random sampling
        source (str): Source file path. Optional
        destination (str): The output filepath. Optional
        force (bool): Whether to force execution if the endpoint already exists.
    """

    def __init__(
        self,
        frac: float,
        uservar: str = "userId",
        itemvar: str = "movieId",
        source: str = None,
        destination: str = None,
        random_state: int = None,
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._frac = frac
        self._uservar = uservar
        self._itemvar = itemvar
        self._random_state = random_state

    @cache
    def execute(self, data: pd.DataFrame = None) -> None:
        """Samples the data"""
        if not self._skip(endpoint=self._destination):
            sample = pd.DataFrame()
            # Get the data. Priority goes to data passed into the method.
            data = data or self._get_data(filepath=self._source)
            # Obtain value counts
            counts = data[self._uservar].value_counts(sort=False).to_frame().reset_index()
            counts.columns = [self._uservar, "count"]
            # Group by user and take random sampling of frac proportion of the user interactions
            for user, ratings in tqdm(data.groupby(self._uservar)):
                # Compute number of ratings to obtain
                n_ratings = int(counts[counts[self._uservar] == user]["count"] * self._frac)
                # Shuffle ratings for the user
                ratings_shuffled = ratings.sample(frac=1, random_state=self._random_state)
                # Take frac proportion of the samples.
                sub_sample = ratings_shuffled[:n_ratings]
                sample = pd.concat([sample, sub_sample], axis=0)

            # Save data if destination is provided.
            self._put_data(filepath=self._destination, data=sample)
            # eh viola!
            return sample
