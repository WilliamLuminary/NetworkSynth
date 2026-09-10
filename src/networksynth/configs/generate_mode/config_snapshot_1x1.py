# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

from .config_snapshot import CONFIG as SNAPSHOT

CONFIG = replace(
    SNAPSHOT,
    SYNTHETIC_FRAME_SIZE=(510, 510),
    SNAPSHOT_INTERVAL=10,
    SYNTHETIC_GRAPH_NUMBER=1,
    SYNTHETIC_NETWORK_NUMBER=1,
    RENDER_BFS_SNAPSHOT=replace(
        SNAPSHOT.RENDER_BFS_SNAPSHOT, node_size=6.0, line_width=3.0
    ),
    RENDER_SYNTHETIC_GRAPH=replace(
        SNAPSHOT.RENDER_SYNTHETIC_GRAPH, node_size=6.0, line_width=3.0
    ),
)
