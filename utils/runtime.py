from __future__ import annotations

import functools
import logging
import time

import numpy as np


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
