#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/sparse.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 3rd 2023 06:31:24 pm                                                #
# Modified   : Friday February 3rd 2023 06:42:21 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Sparse Data Module"""
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse


# ------------------------------------------------------------------------------------------------ #
def df_to_sparse_tensor(df: pd.DataFrame) -> tf.SparseTensor:
    coo = scipy.sparse.coo_array(df.values)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


# ------------------------------------------------------------------------------------------------ #
def pprint_sparse_tensor(st):
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += "\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + "}>"
