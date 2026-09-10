# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

from .config_snapshot import CONFIG as SNAPSHOT

CONFIG = replace(
    SNAPSHOT,
    SYNTHETIC_FRAME_SIZE=(1530, 1530),
    SNAPSHOT_INTERVAL=150,
    SELECT_BEST=3,
    SYNTHETIC_GRAPH_NUMBER=3,
    SYNTHETIC_NETWORK_NUMBER=20,
    ERROR_CHECKER="none",
    ERROR_TOLERANCE=0,
    RENDER_BFS_SNAPSHOT=replace(
        SNAPSHOT.RENDER_BFS_SNAPSHOT, node_size=1.5, line_width=1.0
    ),
    RENDER_SYNTHETIC_GRAPH=replace(
        SNAPSHOT.RENDER_SYNTHETIC_GRAPH, node_size=1.5, line_width=1.0
    ),
)
