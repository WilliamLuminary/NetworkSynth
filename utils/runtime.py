from __future__ import annotations

import functools
import logging
import time

import numpy as np


def apply_seed(seed: int | None) -> None:
    """Seed both RNGs used by generation, or leave them alone if *seed* is None.

    Generation draws from ``random`` (``_graph_node``, ``graph_generator``,
    ``mapper``) and from ``np.random`` (``_graph_node``, ``mapper``), so both
    must be seeded for a run to be reproducible.  Call this at the top of every
    worker, using that worker's derived seed.
    """
    if seed is None:
        return

    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))


def log_memory(label: str, enabled: bool) -> None:
    """Log current process RSS memory usage when *enabled*.

    The gate is a parameter rather than a config read so this stays usable from
    worker processes, where a mutated ``BaseConfig`` is not visible.
    """
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
    """The multiprocessing context every pool and process should use.

    ``spawn`` rather than the platform default. A forked child inherits the
    parent's memory image, including an OpenMP runtime that believes it owns one
    thread per core while the child has one; it then waits on locks held by
    threads that do not exist, and on a many-core host a quality-gated run never
    finished. Every pool worker also pins networkit to one thread, which fixes
    that symptom — this removes the mechanism.

    Passed per pool rather than set with ``set_start_method``, which is
    process-global: a test importing an entry point must not reconfigure the
    whole session, and tests that fork while production spawns is exactly how a
    spawn-only bug (an unpicklable run-time config class) stayed hidden.

    Safe because no pipeline depends on inherited state: parameters travel as an
    immutable ``SynthParams`` and the output location as a ``RunPaths``.
    """
    import multiprocessing as mp

    return mp.get_context("spawn")
