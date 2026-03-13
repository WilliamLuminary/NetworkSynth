# src/configs/generate_mode/config_snapshot_1x1.py
"""
1×1 BFS snapshot config for sample_A.

Generates at original resolution (510×510) with snapshot every 10 nodes.
"""
from typing import Tuple

from .config_snapshot import SnapshotConfig


class Snapshot1x1Config(SnapshotConfig):
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    SNAPSHOT_INTERVAL = 10
    SELECT_BEST = 0
    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    PLOT_STYLE: dict = {
        "dpi": 300,
        "node_size": 6.0,
        "line_width": 3.0,
    }
