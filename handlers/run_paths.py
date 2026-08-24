from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from utils import tagged

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:

    root: str

    def for_dataset(self, dataset_id) -> str:
        return os.path.join(self.root, dataset_id.path)


def create_run_paths(config) -> RunPaths:
    """Create this run's output root and return it as a value.

    A fresh timestamped directory, with ``latest_result`` repointed at it.
    """
    from .saver import _ensure_directory, _time_id, _update_soft_link

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
