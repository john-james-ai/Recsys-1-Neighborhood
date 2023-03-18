#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/services/sparse.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday March 12th 2023 03:03:02 am                                                  #
# Modified   : Wednesday March 15th 2023 10:03:23 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sparse Services Matrix"""
import pandas as pd

# ------------------------------------------------------------------------------------------------ #


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def get_element(matrix, row, col):
    rows, cols = matrix.nonzero()
    d = {"row": rows, "col": cols, "data": matrix.data}
    df = pd.DataFrame(data=d)
    element = df[(df["row"] == row) & (df["col"] == col)]["data"].values[0]
    return element
