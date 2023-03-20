#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /recsys/__main__.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 29th 2023 09:08:36 am                                                #
# Modified   : Monday March 20th 2023 03:31:43 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from recsys.container import Recsys  # pragma: no cover
import logging

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
def wireup():  # pragma: no cover
    container = Recsys()
    container.init_resources()
    container.wire(modules=[__name__, "recsys.asset.centre"])


# ------------------------------------------------------------------------------------------------ #
def main():  # pragma: no cover
    wireup()


# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":  # pragma: no cover
    main()
