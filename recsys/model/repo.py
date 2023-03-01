#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/model/repo.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 06:55:03 am                                                #
# Modified   : Wednesday March 1st 2023 06:55:58 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""ModelAsset Repo"""
from recsys.assets.base import Asset
from recsys.assets.model import ModelAsset
from recsys.repo.asset import AssetRepo
from recsys.persistence.odb import ObjectDB


# ------------------------------------------------------------------------------------------------ #
class ModelAssetRepo(AssetRepo):
    """Repository for Data Assets"""

    def __init__(self, database: ObjectDB, asset_type: type[Asset] = ModelAsset) -> None:
        super().__init__(database=database, asset_type=asset_type)
