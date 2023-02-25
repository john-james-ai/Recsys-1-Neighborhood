#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/sampling.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:51:25 pm                                            #
# Modified   : Friday February 24th 2023 11:08:39 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sampling Operator Module"""
from tqdm import tqdm
import pandas as pd
import numpy as np

from recsys.operator.base import Operator


# ------------------------------------------------------------------------------------------------ #
class UserRandomSampling(Operator):
    """Samples users at random until the target sample size is obtained.

    Args:
        source (str): Filepath of source file
        destination (str): Filepath of sample file
        frac (str): The fraction of the original dataset to retain
        uservar (str): The name of the column containing the user identifier.
        itemvar (str): The name of the column containing the item identifier.
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
        force: bool = False,
        random_state: int = None,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._frac = frac
        self._uservar = uservar
        self._itemvar = itemvar
        self._random_state = random_state

    def run(self, *args, **kwargs) -> None:
        """Samples the data"""
        data = self._fio.read(self._source)

        sample = pd.DataFrame()

        rng = np.random.default_rng(self._random_state)
        size = int(data.shape[0] * self._frac)
        users = data[self._uservar].unique()

        with tqdm(total=size) as pbar:
            for user in rng.shuffle(users):
                if sample.shape[0] >= size:
                    break
                interactions = data[data[self._uservar] == user]
                sample = pd.concat([sample, interactions], axis=0)
                pbar.update(interactions.shape[0])

        self._fio.write(filepath=self._destination, data=sample)
