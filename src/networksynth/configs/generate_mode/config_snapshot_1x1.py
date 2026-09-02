# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace
from typing import Tuple

from .config_snapshot import SnapshotConfig


class Snapshot1x1Config(SnapshotConfig):
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    SNAPSHOT_INTERVAL = 10
    SELECT_BEST = 0
    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    RENDER_BFS_SNAPSHOT = replace(
        SnapshotConfig.RENDER_BFS_SNAPSHOT, node_size=6.0, line_width=3.0
    )
    RENDER_SYNTHETIC_GRAPH = replace(
        SnapshotConfig.RENDER_SYNTHETIC_GRAPH, node_size=6.0, line_width=3.0
    )
