#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/persistence/exceptions.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 28th 2023 02:30:33 pm                                              #
# Modified   : Tuesday February 28th 2023 04:09:13 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #


class ObjectNotFoundError(Exception):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)


class ObjectExistsError(Exception):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)


class ObjectDBEmpty(Exception):  # pragma: no cover
    def __init__(self, msg) -> None:
        super().__init__(msg)
