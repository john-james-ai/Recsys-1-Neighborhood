#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/workflow/schema.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 05:18:14 pm                                                #
# Modified   : Sunday March 19th 2023 04:13:29 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Workflow Schema Module: Defines Object Encapsulating Pipeline Input and Output Requirements"""
from dataclasses import dataclass
from datetime import datetime


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataSchema:
    id: str  # The name and the date it was created in <name_MMDDYYYY> format
    name: str  # The name of the data object.
    desc: str  # The desc of the Data object
    filename: str  # The filename of the Data object
    type: str  # The class of the object
    size: float  # The proportion of the total size if this is part of a ModelDataSchema
    created: datetime  # Datetime the model data schema object was created
    accessed: datetime  # Datetime the model data schema object was last accessed
    modified: datetime  # Datetime the model data schema object was last modified


@dataclass
class ModelDataSchema:
    id: str  # The name and the date it was created in <name_MMDDYYYY> format
    name: str  # The name of the model data schema
    desc: str  # Describe the schema in terms of its purpose stage, state
    location: str  # The directory containing the data
    full_train: DataSchema  # The full training set
    train: DataSchema  # The train DataSchema object
    validation: DataSchema  # The validation set schema
    test: DataSchema  # The test set schema
    created: datetime  # Datetime the model data schema object was created
    accessed: datetime  # Datetime the model data schema object was last accessed
    modified: datetime  # Datetime the model data schema object was last modified
