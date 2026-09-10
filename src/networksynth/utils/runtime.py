# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import functools
import logging
import os
import time

import numpy as np


def worker_count(num_tasks: int | None = None, ceiling: int | None = None) -> int:
    import psutil

    usable = os.process_cpu_count() or os.cpu_count() or 1
    workers = min(psutil.cpu_count(logical=False) or max(1, usable // 2), usable)
    if ceiling is not None:
        workers = min(workers, ceiling)

    footprint = psutil.Process().memory_info().rss
    if footprint > 0:
        workers = min(workers, int(psutil.virtual_memory().available // footprint))

    if num_tasks is not None:
        workers = min(workers, num_tasks)
    return max(1, workers)


def apply_seed(seed: int | None) -> None:
    if seed is None:
        return

    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))


def log_memory(label: str, enabled: bool) -> None:
    from .run_log import tagged

    if not enabled:
        return

    import psutil

    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024**3)
    logging.getLogger(__name__).info(
        f"{label}: RSS = {rss_gb:.2f} GB",
        extra=tagged("MEMORY"),
    )


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(
            "Time taken by func %r: %.2f seconds",
            func.__name__,
            end - start,
        )
        return result

    return wrapper


def spawn_context():
    import multiprocessing as mp

    return mp.get_context("spawn")
