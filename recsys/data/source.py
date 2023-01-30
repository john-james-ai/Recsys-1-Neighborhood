#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/data/source.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning-udemy                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 05:31:48 pm                                                #
# Modified   : Monday January 30th 2023 06:37:41 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from datetime import datetime
from tqdm import tqdm
from typing import List, Union
from urllib.parse import urlsplit
import requests

from recsys.data.base import DataSource


# ------------------------------------------------------------------------------------------------ #
class WebDataSource(DataSource):
    """Web data source

    Args:
        name (str): Name of the resource
        description (str): Description of the resource
        website (str): The website for the data source
        urls (list): List of URL objects
    """

    def __init__(
        self,
        name: str,
        website: str,
        urls: Union[str, List[str]],
        description: str = None,
    ) -> None:
        super().__init__(name=name, description=description)

        self._website = website

        if isinstance(urls, list):
            self._urls = urls
        elif isinstance(urls, str):
            self._urls = [urls]
        else:
            msg = "Urls must be a list or a string."
            self._logger.error(msg)
            raise TypeError(msg)

        self._download_time = None
        self._size = self._get_size()
        self._exists = self._size > 0

    # -------------------------------------------------------------------------------------------- #
    @property
    def website(self) -> int:
        return self._website

    @property
    def exists(self) -> str:
        return self._exists

    @property
    def size(self) -> int:
        return self._size

    @property
    def download_time(self) -> str:
        return self._download_time

    # -------------------------------------------------------------------------------------------- #
    def download(self, destination: str, force: bool = False) -> None:
        """Downloads the resource

        Args:
            destination (str): Directory to which the resource will be downloaded.
            force (bool): Indicates whether to force download if the file
                already exists. Defaults to False.
        """
        start = datetime.now()
        self._size = 0

        for url in tqdm(self._urls):
            self._download(url=url, destination=destination, force=force)

        end = datetime.now()
        self._download_time = (end - start).total_seconds() * 0.001

    def _download(self, url: str, destination: str, force: bool = False) -> None:

        # Set destination filepath
        filepath = os.path.join(destination, self._extract_filename(url))

        if force or not os.path.exists(filepath):
            # Ensure the directory exists
            os.makedirs(destination, exist_ok=True)
            # Obtain the a url response object
            resp = self._request(url=url, stream=True)
            # Get total file size for the progress bar
            total = int(resp.headers.get("content-length", 0))
            # Open the destination file and tqdm progress bar
            with open(filepath, "wb") as file, tqdm(
                desc=filepath,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                # Download data in chunks and update progressbar.
                self._logger.debug(f"Downloading {url} to {filepath}")
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        else:
            self._logger.info(f"Download of {url} was skipped. File already exists.")

    # -------------------------------------------------------------------------------------------- #
    def _request(self, url: str, stream: bool = False) -> requests.Response:

        try:
            resp = requests.get(url, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            self._logger.error(f"HTTP Error accessing {url}\n{e}")
            raise
        except requests.exceptions.ConnectionError as e:
            self._logger.error(f"Connection Error accessing {url}\n{e}")
            raise
        except requests.exceptions.Timeout as e:
            self._logger.error(f"Connection Timeout Error accessing {url}\n{e}")
            raise
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Request Error accessing {url}\n{e}")
            raise
        except requests.exceptions.MissingSchema() as e:
            self._logger.error(f"Invalid URL {url}\n{e}")
            raise

    # -------------------------------------------------------------------------------------------- #
    def _get_size(self) -> int:
        """Obtains the total size of the data source in bytes"""
        size = 0
        for url in self._urls:
            response = self._request(url)
            size += int(response.headers.get("content-length", 0))
        return size

    # -------------------------------------------------------------------------------------------- #
    def _extract_filename(self, url: str) -> str:
        filepath = urlsplit(url).path
        return filepath.rsplit("/", 1)[1]
