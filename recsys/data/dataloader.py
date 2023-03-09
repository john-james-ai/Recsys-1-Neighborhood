#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Recommender Systems in Python 1: Neighborhood Methods                               #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /recsys/data/dataloader.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/recsys-01-collaborative-filtering                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 6th 2023 12:00:52 am                                                   #
# Modified   : Monday March 6th 2023 02:33:12 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DataLoader Module."""
import numpy as np
import multiprocessing
import logging
import queue
from itertools import cycle
from typing import Any

from recsys.services.io import IOService


# ------------------------------------------------------------------------------------------------ #
def default_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (int, float)):
        return np.array(batch)
    if isinstance(batch[0], (list, tuple)):
        return tuple(default_collate(var) for var in zip(*batch))
    else:
        return batch


class DataLoader:
    def __init__(
        self,
        filepath: str = None,
        dataset: Any = None,
        batch_size: int = 1024,
        collate_fn=default_collate,
    ):
        self._filepath = filepath
        self._index = 0
        self._dataset = dataset or IOService.read(filepath)
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._logger = logging.getLogger(
            f"{self.__module__}.{self.__class__.__name__}",
        )

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._dataset):
            raise StopIteration
        batch_size = min(len(self._dataset) - self._index, self._batch_size)
        return self.get(batch_size)

    # fmt: off
    def get(self, batch_size: int):
        item = self._dataset.iloc[
            (self._index * batch_size): (self._index + 1) * batch_size
        ]  # noqa E203
        self._index += 1
        return item


# fmt: on

# fmt: off
def worker_fn(dataset, index_queue, output_queue, batch_size):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        data = dataset.iloc[(index * batch_size): (index + 1) * batch_size]
        output_queue.put((index, data))
# fmt: on


class TabularDataLoader(DataLoader):
    def __init__(
        self,
        filepath: str = None,
        dataset: Any = None,
        batch_size: int = 1024,
        num_workers: int = 4,
        prefetch_batches: int = 2,
        collate_fn=default_collate,
    ):
        super().__init__(
            filepath=filepath, dataset=dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        self._num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self._output_queue = multiprocessing.Queue()
        self._index_queues = []
        self._workers = []
        self.worker_cycle = cycle(range(num_workers))
        self._cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()

            worker = multiprocessing.Process(
                target=worker_fn,
                args=(self._dataset, index_queue, self._output_queue, self._batch_size),
            )
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self):
        while (
            self.prefetch_index * self._batch_size < len(self._dataset)
            and self.prefetch_index < self._index + 2 * self._num_workers * self._batch_size
        ):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self._index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def __iter__(self):
        self._index = 0
        self._cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def get(self, batch_size):
        self.prefetch()
        if self._index in self._cache:
            item = self._cache[self._index]
            del self._cache[self._index]
        else:
            while True:
                try:
                    (index, data) = self._output_queue.get(timeout=0)

                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self._index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self._cache[index] = data

        self._index += 1
        return item

    def __del__(self):
        try:
            for i, w in enumerate(self._workers):
                self._index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self._index_queues:
                q.cancel_join_thread()
                q.close()
            self._output_queue.cancel_join_thread()
            self._output_queue.close()
        finally:
            for w in self._workers:
                if w.is_alive():
                    w.terminate()
