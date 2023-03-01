#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/exceptions/database.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 02:30:33 pm                                              #
# Modified   : Tuesday February 28th 2023 11:13:38 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.exceptions.base import RecsysException


class ObjectNotFoundError(RecsysException):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)


class ObjectExistsError(RecsysException):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)


class ObjectDBEmpty(RecsysException):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)
