#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/data/reindex.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 3rd 2023 02:45:26 pm                                                   #
# Modified   : Saturday March 4th 2023 06:07:01 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os

import pandas as pd

from recsys import Operator


# ------------------------------------------------------------------------------------------------ #
#                                TEMPORAL TRAIN/TEST SPLIT                                         #
# ------------------------------------------------------------------------------------------------ #
class Reindex(Operator):
    """Converts user and movie ids to sequential indices.

    Args:
        source (str): Source file path. Optional
        destination (str): The output directory. Optional)
        uservar (str): Column  containing user id
        itemvar (str): Column containing item id
        useridx (str): Column containing the sequential user id
        itemidx (str): Column containing the sequential item id
        force (bool): Whether to overwrite existing data if it already exists.
    """

    def __init__(
        self,
        source: str = None,
        destination: str = None,
        uservar: str = "userId",
        itemvar: str = "movieId",
        useridx: str = "useridx",
        itemidx: str = "itemidx",
        force: bool = False,
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._uservar = uservar
        self._itemvar = itemvar
        self._useridx = useridx
        self._itemidx = itemidx

    def execute(self, data: pd.DataFrame = None) -> None:
        """Performs the train test split."""
        if not self._skip(endpoint=self._destination):

            data = data or self._get_data(filepath=self._source)

            try:

                data_sorted = data.sort_values(by=[self._timestamp_var], ascending=True)
                train_size = int(self._train_size * data.shape[0])

                train = data_sorted[0:train_size]
                test = data_sorted[train_size:]

                self._put_data(filepath=self._train_filepath, data=train)
                self._put_data(filepath=self._test_filepath, data=test)

                result = {"train": train, "test": test}
                return result
            except KeyError:
                msg = f"Column {self._timestamp_var} was not found."
                self._logger.error(msg)
                raise
