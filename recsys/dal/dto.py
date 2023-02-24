#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dal/dto.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 06:35:03 am                                            #
# Modified   : Wednesday February 22nd 2023 10:58:15 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Transfer Object Module"""
from dataclasses import dataclass

from recsys.dal.base import DTO


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetDTO(DTO):
    id: int = None
    name: str = None
    type: str = None
    description: str = None
    lab: str = None
    filepath: str = None
    stage: str = None
    rows: int = None
    cols: int = None
    n_users: int = None
    n_items: int = None
    size: int = None
    matrix_size: int = None
    memory_mb: int = None
    cost: int = None
    sparsity: int = None
    density: str = None
    created: str = None
