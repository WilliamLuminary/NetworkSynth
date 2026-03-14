# src/configs/generate_mode/config_snapshot_3x3.py
"""
3×3 BFS snapshot config for sample_A.

Generates at 3× original resolution (1530×1530) with snapshot every 50 nodes.
"""
from typing import Tuple

from .config_snapshot import SnapshotConfig


class Snapshot3x3Config(SnapshotConfig):
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1530, 1530)

    SNAPSHOT_INTERVAL = 150
    SELECT_BEST = 3
    SYNTHETIC_GRAPH_NUMBER = 3
    SYNTHETIC_NETWORK_NUMBER = 20
    ERROR_TOLERANCE = 0

    PLOT_STYLE: dict = {
        "dpi": 300,
        "node_size": 1.5,
        "line_width": 1.0,
    }
