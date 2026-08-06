# src/handlers/run_paths.py
"""Where a run writes its output.

``Saver`` used to answer this with a class attribute set by an ``initialize()``
classmethod.  That made the answer global: two runs could not coexist in one
process, a forked child inherited the parent's directory by accident rather than
by design, and tests had to reset the class attribute between cases.

A run's output location is just path arithmetic, so it is a *value* here.
``RunPaths`` is frozen and picklable, which means many runs are simply many
values, and passing one to a worker process works the same way on ``fork`` and
``spawn``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from configs.base_config import tagged

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:
    """The output root for one run, plus per-dataset resolution."""

    root: str

    def for_dataset(self, dataset_id) -> str:
        """Directory for *dataset_id* inside this run's root."""
        return os.path.join(self.root, dataset_id.path)


def create_run_paths(config, result_dir: str | None = None) -> RunPaths:
    """Create this run's output root and return it as a value.

    When *result_dir* is given the run writes directly there (analysis mode
    re-reading an existing result set).  Otherwise a fresh timestamped
    directory is created and ``latest_result`` is repointed at it.
    """
    from .saver import _ensure_directory, _time_id, _update_soft_link

    if result_dir:
        return RunPaths(root=os.path.join(config.BASE_OUTPUT_PATH, result_dir))

    root = os.path.join(
        config.BASE_OUTPUT_PATH,
        f"{config.OUTPUT_DENOTE}_results_{_time_id()}_{config.RUN_ID}",
    )
    if config.DISABLE_SAVING:
        # Still return the value (callers need something to pass down) but do
        # not create the directory or repoint `latest_result`: the old
        # Saver.initialize() skipped both when saving was off, and a disabled
        # run should leave no trace.
        return RunPaths(root=root)

    _ensure_directory(root)
    logger.info(f"Base Output directory: {root}", extra=tagged("IO"))
    _update_soft_link(os.path.join(config.BASE_OUTPUT_PATH, "latest_result"), root)
    return RunPaths(root=root)
