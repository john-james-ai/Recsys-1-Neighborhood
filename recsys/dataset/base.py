#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataset/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday March 17th 2023 03:33:14 pm                                                  #
# Modified   : Friday March 17th 2023 08:01:09 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Dataset(ABC):
    """Defines the interface for recommender data sources"""

    @abstractmethod
    def fetch_data(self) -> None:
        """Downloads and extracts the data if it is not already present"""
