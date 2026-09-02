# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from networksynth.utils import tagged

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:

    root: str

    def for_dataset(self, dataset_id) -> str:
        return os.path.join(self.root, dataset_id.path)


def create_run_paths(config) -> RunPaths:
    from .saver import _ensure_directory, _time_id, _update_soft_link

    root = os.path.join(
        config.BASE_OUTPUT_PATH,
        f"{config.OUTPUT_DENOTE}_results_{_time_id()}_{config.RUN_ID}",
    )
    if config.DISABLE_SAVING:
        return RunPaths(root=root)

    _ensure_directory(root)
    logger.info(f"Base Output directory: {root}", extra=tagged("IO"))
    _update_soft_link(os.path.join(config.BASE_OUTPUT_PATH, "latest_result"), root)
    return RunPaths(root=root)
