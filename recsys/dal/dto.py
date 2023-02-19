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
# Created    : Saturday February 18th 2023 06:03:34 pm                                             #
# Modified   : Sunday February 19th 2023 05:01:15 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

from recsys.dal.base import DTO

# ------------------------------------------------------------------------------------------------ #


@dataclass(eq=False)
class DatasetDTO(DTO):
    id: int
    type: str
    name: str
    description: str
    memory: float
    cost: float

    def __eq__(self, other) -> bool:
        if isinstance(other, DatasetDTO):
            return (
                self.type == other.type
                and self.name == other.name
                and self.description == other.description
                and self.memory == other.memory
                and self.cost == other.cost
            )
        else:
            return False
