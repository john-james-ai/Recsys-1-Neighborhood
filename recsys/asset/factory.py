#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/asset/factory.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 20th 2023 07:25:02 am                                                  #
# Modified   : Monday March 20th 2023 08:27:58 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Asset Factory Module"""
import logging

from recsys.datasource.movielens import MovieLens10M, MovieLens1M, MovieLens25M
from recsys.dataset.movielens import MovieLens
from recsys.asset.base import Asset


class AssetFactory:

    __asset_types = {
        "movielens1m": MovieLens1M,
        "movielens10m": MovieLens10M,
        "movielens25m": MovieLens25M,
        "movielens": MovieLens,
    }

    @classmethod
    def build(cls, asset_schema: dict, **kwargs) -> Asset:
        """Constructs an asset according to the schema

        Args:
            asset_schema (dict): Dictionary with two primary keys:
             - asset_type with a string value
             - params with a dictionary value including parameters.
            kwargs (dict): Other parameters which were only available at runtime.

        """
        cls._logger = logging.getLogger(
            f"{cls.__module__}.{cls.__class__.__name__}",
        )
        try:
            asset = AssetFactory.__asset_types[asset_schema["asset_type"].lower()]
        except KeyError:
            msg = "Asset schema is malformed. Must have a key of 'asset_type'"
            cls._logger.error(msg)
            raise KeyError(msg)
        try:
            params = asset_schema["params"]
        except KeyError:
            msg = "Asset schema is malformed. Must have a key of 'params'"
            cls._logger.error(msg)
            raise KeyError(msg)

        # Add dictionary containing additional parameters which become available
        # at runtime, to the params dictionary.
        params = params.update(kwargs)
        return asset(**params)
