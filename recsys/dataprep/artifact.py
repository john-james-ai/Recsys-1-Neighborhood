#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/dataprep/artifact.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 11:10:07 am                                                #
# Modified   : Saturday March 18th 2023 11:10:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import mlflow


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Artifact:
    isfile: bool  # Indicates whether the artifact is a file or a directory
    path: str  # The filepath or directory containing the artifacts
    uripath: str  # The directory within the artifact store to place the artifact

    def log(self) -> None:
        if self.isfile:
            mlflow.log_artifact(local_path=self.path, artifact_path=self.uripath)
        else:
            mlflow.log_artifacts(local_path=self.path, artifact_path=self.uripath)
