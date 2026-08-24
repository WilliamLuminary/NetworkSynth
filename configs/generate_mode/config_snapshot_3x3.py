from dataclasses import replace
from typing import Tuple

from .config_snapshot import SnapshotConfig


class Snapshot3x3Config(SnapshotConfig):
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1530, 1530)

    SNAPSHOT_INTERVAL = 150
    SELECT_BEST = 3
    SYNTHETIC_GRAPH_NUMBER = 3
    SYNTHETIC_NETWORK_NUMBER = 20
    ERROR_CHECKER = "none"
    ERROR_TOLERANCE = 0

    RENDER_BFS_SNAPSHOT = replace(
        SnapshotConfig.RENDER_BFS_SNAPSHOT, node_size=1.5, line_width=1.0
    )
    RENDER_SYNTHETIC_GRAPH = replace(
        SnapshotConfig.RENDER_SYNTHETIC_GRAPH, node_size=1.5, line_width=1.0
    )
