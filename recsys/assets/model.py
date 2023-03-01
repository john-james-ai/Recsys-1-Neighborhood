#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/assets/model.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday March 1st 2023 12:38:34 am                                                #
# Modified   : Wednesday March 1st 2023 04:55:36 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Any

from recsys import Asset


# ------------------------------------------------------------------------------------------------ #
#                                  MODEL CLASS                                                     #
# ------------------------------------------------------------------------------------------------ #
class ModelAsset(Asset):  # pragma: no cover
    """Base class for models.
    Args:
        name (str): Name of the model
        description (str): Description of the model
        model (Any): The model object.
    """

    def __init__(self, name: str, description: str, model: Any) -> None:
        super().__init__(name=name, description=description)
        self._model = model

    @property
    def model(self) -> Any:
        return self._model
