#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/datasource/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 03:33:14 pm                                                  #
# Modified   : Monday March 20th 2023 09:51:22 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataSource(ABC):
    """Defines the interface for recommender data sources"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fetch_data(self) -> None:
        """Downloads and extracts the data if it is not already present"""
