#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/io/zip.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 30th 2023 01:55:53 am                                                #
# Modified   : Monday January 30th 2023 06:13:14 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Zip utility module"""
from zipfile import ZipFile
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------ #
def extractzip(source: str, destination: str) -> None:
    """Extracts files from a source archive

    Args:
        source (str): Filepath of the zip archive file
        destination (str): The directory into which the files will be stored
    """
    with ZipFile(source, "r") as zf:
        files = zf.namelist()
        for file in tqdm(files, total=len(files)):
            zf.extract(member=file, path=destination)
