#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems Lab: Towards State-of-the-Art                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /jbook/build.sh                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-lab                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday March 18th 2023 10:58:04 pm                                                #
# Modified   : Saturday March 18th 2023 10:58:05 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
#!/bin/sh
# Delete prior build
#echo "Removing prior jupyter-book build artifacts..."
#rm -r jbook/_build/
#Prepare notebook display customizations
echo "Preparing notebook tags..."
python3 jbook/prep_notebooks.py
# Rebuilds the book
echo "Building book..."
jb build jbook/
# Commit book to gh-pages
echo "Committing changes to github pages..."
ghp-import -o -n -p -f jbook/_build/html