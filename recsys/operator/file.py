#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/operator/file.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday February 22nd 2023 07:35:10 pm                                            #
# Modified   : Wednesday February 22nd 2023 07:54:12 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
import urllib
from zipfile import ZipFile

from recsys.operator.base import Operator


# ------------------------------------------------------------------------------------------------ #
#                                  DOWNLOAD EXTRACTOR ZIP                                          #
# ------------------------------------------------------------------------------------------------ #
class DownloadZipExtract(Operator):
    """Downloads and extracts DataSource files to a destination directory.
    Args:
        source (str): The source URL from which the file will be downloaded
        destination (str): A directory into which the ZipFile contents will be extracted.
    """

    __tempfile = "/tmp/tempfile.zip"

    def __init__(self, source: str, destination: str) -> None:
        super().__init__()
        self._source = source
        self._destination = destination

    def execute(self, *args, **kwargs) -> None:
        """Downloads and extracts the DataSource."""
        zipresp = urllib.request.urlopen(self._source)
        # Create a new file on the hard drive
        tempzip = open(DownloadZipExtract.__tempfile, "wb")
        # Write the contents of the downloaded file into the new file
        tempzip.write(zipresp.read())
        # Close the newly-created file
        tempzip.close()
        # Re-open the newly-created file with ZipFile()
        zf = ZipFile(DownloadZipExtract.__tempfile)
        # Extract its contents into <destination_path>
        # note that extractall will automatically create the path
        zf.extractall(path=self._destination)
        # close the ZipFile instance
        zf.close()
