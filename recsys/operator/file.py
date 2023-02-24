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
# Modified   : Thursday February 23rd 2023 01:57:28 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Mover Module"""
import os
from abc import abstractmethod
import urllib
from zipfile import ZipFile

from recsys.operator.base import Operator
from recsys.operator.system import eventlog


# ------------------------------------------------------------------------------------------------ #
#                                      DOWNLOADER ABC                                              #
# ------------------------------------------------------------------------------------------------ #
class DownloaderABC(Operator):
    def __init__(self, url: str, destination: str) -> None:
        super().__init__()
        self._url = url
        self._destination = destination

    @abstractmethod
    def execute(self) -> None:
        """Downloads the data to the destination directory."""


# ------------------------------------------------------------------------------------------------ #
#                                      DOWNLOADERFILE                                              #
# ------------------------------------------------------------------------------------------------ #
class DownloaderFile(DownloaderABC):
    """Downloads an uncompressed file from a website.
    Args:
        url (str): The URL to the web resource
        destination (str): A directory into which the source file will be stored. The destination
            file will have the same name as the source file.
    """

    def __init__(self, url: str, destination: str) -> None:
        super().__init__(url=url, destination=destination)

    def execute(self) -> None:
        """Downloads a file from a remote source."""
        try:
            filename = os.path.basename(self._url)
            destination = os.path.join(self._destination, filename)
            _ = urllib.request.urlretrieve(url=self._url, filename=destination)
        except IsADirectoryError:
            msg = "The destination parameter is a directory. For download, this must be a path to a file."
            self._logger.error(msg)
            raise IsADirectoryError(msg)


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

    def __init__(self, source: str, download: str, extract: str, force: bool = False) -> None:
        super().__init__()
        self._source = source
        self._downloadd = download
        self._extract = extract

    @eventlog
    def execute(self, *args, **kwargs) -> None:
        """Downloads and extracts the DataSource."""

        if self.abort(target_directory=self._extract):
            return
        else:
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

    def abort(self, target_directory: str, force: bool = False) -> bool:
        """Skip if Force is False and target already exists.

        Args:
            target (str): The destinatoion directory.
        """
        if not force and len(os.listdir(target_directory) > 0):
            # Assumed to already exist and operation is aborted.
            return True
        else:
            return False
