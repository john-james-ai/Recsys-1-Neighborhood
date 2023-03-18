#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/dataset/sampling.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 05:51:25 pm                                            #
# Modified   : Friday March 17th 2023 03:00:23 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sampling Operator Module"""
from tqdm import tqdm
import pandas as pd

from recsys import Operator, Artifact


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
        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
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
            # Announce
            self._announce()
            # eh viola!
            return sample

    def _announce(self) -> None:
        if self._source is None and self._destination is None:
            self._logger.info(f"Created a {int(self._frac*100)}% sample of dataset.")
        elif self._source is None:
            self._logger.info(
                f"Created a {int(self._frac*100)}% sample of dataset at {self._destination}."
            )
        elif self._destination is None:
            self._logger.info(f"Created a {int(self._frac*100)}% sample of dataset {self._source}.")
        else:
            self._logger.info(
                f"Created a {int(self._frac*100)}% sample of dataset {self._source} at {self._destination}."
            )


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
        source (str): Source file path.
        destination (str): The output filepath.
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
        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
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


# ------------------------------------------------------------------------------------------------ #
class TemporalInteractionThresholdSampling(Operator):
    """Samples the most recent umax user interactions and imax item interactions

    Args:
        source (str): Source file path.
        destination (str): The output filepath.
        umax (int): Maximum number of interactions per user
        imax (int): Maximum number of interactions per item.
        force (bool): Whether to force execution if the endpoint already exists.

    """

    def __init__(
        self, source: str, destination: str, umax: int = 1000, imax: int = 1000, force: bool = False
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._umax = umax
        self._imax = imax

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
        """Controls the sampling process."""

        if not self._skip(endpoint=self._destination):
            # Obtain data from the source
            data = data or self._get_data(filepath=self._source)

            # Sample the most recent user and item interactions, up to the respective thresholds
            data = self._sample_interactions(data=data, by="userId", threshold=self._umax)
            data = self._sample_interactions(data=data, by="movieId", threshold=self._imax)

            data.drop(columns=["interactions"], inplace=True)

            self._put_data(filepath=self._destination, data=data)

            return data

    def _sample_interactions(self, data: pd.DataFrame, by: str, threshold: int) -> pd.DataFrame:
        """Samples the interactions by user or item"""

        # Groupby user or item, sort by timestamp descending within groups, and count the interactions.
        data["interactions"] = (
            data.sort_values(by="timestamp", ascending=False)
            .groupby(by=by, group_keys=False)
            .cumcount()
            + 1
        )

        # Retain the most recent interactions up to the  threshold
        data = data[data["interactions"] <= threshold]

        return data


# ------------------------------------------------------------------------------------------------ #
class RandomTemporalInteractionThresholdSampling(Operator):
    """Interactions are processed by timestamp and randomly ejected when the threshold is met.

    Interactions are processed in order. If the threshold is met as a consequence of a new
    interaction, a random interaction from the user or item history is ejected and replaced
    by the new interaction [1]_.

    .. [1] DOI 10.1145/3335783.3335784

    Args:
        source (str): Source file path.
        destination (str): The output filepath.
        umax (int): Maximum number of interactions per user
        imax (int): Maximum number of interactions per item.
        force (bool): Whether to force execution if the endpoint already exists.

    """

    def __init__(
        self, source: str, destination: str, umax: int = 1000, imax: int = 1000, force: bool = False
    ) -> None:
        super().__init__(source=source, destination=destination, force=force)
        self._umax = umax
        self._imax = imax
        self._interactions_cut = 0

        self._artifact = Artifact(isfile=True, path=self._destination, uripath="data")

    def execute(self, data: pd.DataFrame = None, context: dict = None) -> None:
        """Controls the sampling process."""

        if not self._skip(endpoint=self._destination):
            # Obtain data from the source
            data = data or self._get_data(filepath=self._source)

            # Sample the most recent user and item interactions, up to the respective thresholds
            data = self._sample_interactions(data=data, by="userId", threshold=self._umax)
            data = self._sample_interactions(data=data, by="movieId", threshold=self._imax)

            data.drop(columns=["interactions"], inplace=True)

            self._put_data(filepath=self._destination, data=data)

            self._logger.debug(f"\nInteractions cut: {self._interactions_cut}")

            return data

    def _sample_interactions(self, data: pd.DataFrame, by: str, threshold: int) -> pd.DataFrame:
        """Obtain the first transactions up to the respective threshold."""

        # Groupby user or item, sort by timestamp ascending within groups, and count the interactions.
        data["interactions"] = (
            data.sort_values(by="timestamp", ascending=True)
            .groupby(by=by, group_keys=False)
            .cumcount()
            + 1
        )

        # Retain the most recent interactions up to the  threshold
        data = data[data["interactions"] <= threshold]
        pending = data[data["interactions"] > threshold]

        if pending.shape[0] > 0:
            data = self._handle_pending(data=data, pending=pending)
        return data

    def _handle_pending(self, by: str, data: pd.DataFrame, pending: pd.DataFrame) -> pd.DataFrame:
        """Existing interactions are randomly withdrawn to accommodate new interactions."""

        # Remove a single interaction at random and replace with new interaction.
        groups = pending.sort_values(by=["timestamp"], ascending=True).groupby(
            by=by, group_keys=False
        )
        for name, ratings in tqdm(groups):
            for index, row in ratings.iterrows():
                row = row.to_frame()
                eviction = data[data[by] == name].sample(n=1, replace=False, axis=0)
                self._interactions_cut += 1
                data = data.drop(eviction.index.values)
                data = pd.concat([data, row.T], axis=0)
        return data
