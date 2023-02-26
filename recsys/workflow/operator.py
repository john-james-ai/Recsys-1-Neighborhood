#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems and Deep Learning in Python                                     #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/workflow/operator.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-deep-learning                               #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 25th 2023 05:43:46 am                                             #
# Modified   : Sunday February 26th 2023 04:47:32 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC, abstractmethod
import logging
from typing import Any, Union


# ------------------------------------------------------------------------------------------------ #
class Operator(ABC):
    """Abstract base class for classes that perform a descrete operation as part of a larger workflow"""

    def __init__(self, *args, **kwargs) -> None:
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Executes the operation"""


# ------------------------------------------------------------------------------------------------ #
class FileOperator(Operator):
    """Base class for operators that manipulate files, i.e. download, compress."""

    def __init__(
        self,
        source: str,
        destination: str,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._source = source
        self._destination = destination
        self._force = force

    @property
    def source(self) -> str:
        return self._source

    @property
    def destination(self) -> str:
        return self._destination

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[Any, None]:
        """Performs the operation."""

    def _skip(self) -> bool:
        """Used to evaluate whether the operation should be skipped."""
        if self._force:
            return False
        elif os.path.isfile(self._destination) and os.path.exists(self._destination):
            self._logger.info(
                f"{self.__class__.__name__} skipped. Destination file {self._destination} already exists. To overwrite set force to True."
            )
            self._status = "skipped"
            return True
        elif (
            os.path.isdir(self._destination)
            and os.path.exists(self._destination)
            and len(os.listdir(self._destination)) > 0
        ):
            self._logger.info(
                f"{self.__class__.__name__} skipped. Destination {self._destination} is not empty. To overwrite set force to True."
            )
            self._status = "skipped"
            return True
        else:
            return False


# ------------------------------------------------------------------------------------------------ #
class SamplingOperator(Operator):
    """Base class for operators that sample data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""


# ------------------------------------------------------------------------------------------------ #
class TransformationOperator(Operator):
    """Base class for operators that transform data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""


# ------------------------------------------------------------------------------------------------ #
class PredictionOperator(Operator):
    """Base class for operators that produce measures or predictions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""


# ------------------------------------------------------------------------------------------------ #
class ModelOperator(Operator):
    """Base class for operators that transform data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""


# ------------------------------------------------------------------------------------------------ #
class ModelSelectionOperator(Operator):
    """Base class for operators that prepare data for modeling according to a preset strategy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""


# ------------------------------------------------------------------------------------------------ #
class ModelEvaluationOperator(Operator):
    """Base class for operators that transform data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def execute(self, data: Any = None) -> Any:
        """Performs the operation on data and returns the data"""
