#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/__init__.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 12:18:02 am                                                #
# Modified   : Tuesday February 28th 2023 06:03:58 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC

from recsys.operator.base import Operator
from recsys.assets.base import Asset
from recsys.persistence.exceptions import ObjectExistsError, ObjectDBEmpty, ObjectNotFoundError
from types import SimpleNamespace

# ------------------------------------------------------------------------------------------------ #
SCHEMA = {
    "userid": "userId",
    "itemid": "movieId",
    "rating": "rating",
    "timestamp": "timestamp",
    "mean_centered_rating_user": "rating_mcu",
    "mean_centered_rating_item": "rating_mci",
    "rating_zscore_user": "rating_zu",
    "rating_zscore_item": "rating_zi",
}
SCHEMA = SimpleNamespace(**SCHEMA)

CACHE_CONFIG = {"expires": "P1W"}
CACHE_CONFIG = SimpleNamespace(**CACHE_CONFIG)
