#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /tests/test_workflow/test_operators/test_file.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 24th 2023 10:11:22 pm                                               #
# Modified   : Sunday February 26th 2023 12:09:12 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging
import shutil

from recsys.io.remote import ZipDownloader
from recsys.io.local import ConvertFile
from recsys.io.compress import ZipExtractor

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.operator
@pytest.mark.file
class TestFileOperators:  # pragma: no cover

    RESULTS = "tests/results/operators/file/"

    # ============================================================================================ #
    # @pytest.mark.skip(reason="Works and takes too long to download")
    def test_setup(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        shutil.rmtree(TestFileOperators.RESULTS, ignore_errors=True)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    # @pytest.mark.skip(reason="Works and takes too long to download")
    def test_download(self, files, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        SOURCE = files.zipfile_url
        DESTINATION = "tests/results/operators/file/zipdownloader/ml-25m.zip"
        dl = ZipDownloader(source=SOURCE, destination=DESTINATION)
        dl.execute()
        assert os.path.exists(DESTINATION)
        assert dl.status == "success"
        assert isinstance(dl.started, datetime)
        assert isinstance(dl.ended, datetime)
        assert isinstance(dl.duration, float)
        dl.execute()
        assert dl.status == "skipped"
        assert isinstance(dl.started, datetime)
        assert isinstance(dl.ended, datetime)
        assert isinstance(dl.duration, float)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_extract(self, files, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        SOURCE = files.zipfile_download
        DESTINATION = "tests/results/operators/file/zipextract/"
        shutil.rmtree(DESTINATION, ignore_errors=True)
        ext = ZipExtractor(source=SOURCE, destination=DESTINATION)
        ext.execute()
        assert len(os.listdir(DESTINATION)) > 0
        assert ext.status == "success"
        assert isinstance(ext.started, datetime)
        assert isinstance(ext.ended, datetime)
        assert isinstance(ext.duration, float)
        ext.execute()
        assert ext.status == "skipped"
        assert isinstance(ext.started, datetime)
        assert isinstance(ext.ended, datetime)
        assert isinstance(ext.duration, float)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_convert(self, files, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        SOURCE = files.ratings_csv
        DESTINATION = "tests/results/operators/file/copy/ratings.pkl"
        shutil.rmtree(os.path.dirname(DESTINATION), ignore_errors=True)
        cp = ConvertFile(source=SOURCE, destination=DESTINATION)
        cp.execute()
        assert os.path.exists(DESTINATION)
        assert cp.status == "success"
        assert isinstance(cp.started, datetime)
        assert isinstance(cp.ended, datetime)
        assert isinstance(cp.duration, float)
        cp.execute()
        assert cp.status == "skipped"
        assert isinstance(cp.started, datetime)
        assert isinstance(cp.ended, datetime)
        assert isinstance(cp.duration, float)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
