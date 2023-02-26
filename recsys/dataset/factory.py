#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/dataset/factory.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 26th 2023 05:16:35 am                                               #
# Modified   : Sunday February 26th 2023 10:02:32 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Dataset Factory Module"""
import os
from abc import ABC, abstractproperty
from dotenv import load_dotenv

from recsys.dataset.base import Dataset

# ------------------------------------------------------------------------------------------------ #
class DatasetFactory(ABC):
    def set_id(self, dataset: Dataset) -> Dataset:
        load_dotenv()
        workspace = os.getenv("WORKSPACE")
        id_key = workspace + "_" + dataset.__class__.__name__.lower()
