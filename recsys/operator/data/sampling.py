#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/sampling.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:51:25 pm                                            #
# Modified   : Sunday February 26th 2023 12:10:39 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sampling Operator Module"""

from recsys import operator
from recsys.workflow.event import event_log


# ------------------------------------------------------------------------------------------------ #
class UserRandomSampling(operator.Operator):
    """Samples users at random until the target sample size is obtained.

    Args:
        source (str): Filepath of source file
        destination (str): Filepath of sample file
        frac (str): The fraction of the original dataset to retain
        uservar (str): The name of the column containing the user identifier.
        itemvar (str): The name of the column containing the item identifier.
        cores (int): The number of cores to run locally.
        force (str): Whether to force execution
        random_state (int): Seed for pseudo random sampling
    """

    def __init__(
        self,
        source: str,
        destination: str,
        frac: float,
        uservar: str = "userId",
        itemvar: str = "movieId",
        cores: int = 8,
        force: bool = False,
        random_state: int = None,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._frac = frac
        self._uservar = uservar
        self._itemvar = itemvar
        self._cores = cores
        self._random_state = random_state

    @event_log
    def run(self, *args, **kwargs) -> None:
        """Samples the data"""
        data = self._fio.read(self._source)

        # Compute the number of samples to return
        size = int(data.shape[0] * self._frac)
        # Obtain value counts
        counts = data[self._uservar].value_counts(sort=False).to_frame().reset_index()
        counts.columns = [self._uservar, "count"]
        # Shuffle and add cumulative sum of counts
        counts = counts.sample(frac=1, random_state=self._random_state)
        counts["cumsum"] = counts["count"].cumsum()
        # Select users where cumulative sum is <= size
        users = counts[counts["cumsum"] <= size][self._uservar].values
        # Select user rows from data.
        sample = data[data[self._uservar].isin(users)]

        # Viola
        self._fio.write(filepath=self._destination, data=sample)
