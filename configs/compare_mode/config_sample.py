# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
import os
from typing import List  # noqa:F401

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MODE = "compare"

    MEASURE_WEIGHTED = False

    DATASETS = [DatasetId("analysis")]

    _RESULTS = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "latest_result", "sample_1")
    ORIGINAL_NETWORKS_PATH = os.path.join(_RESULTS, "original")
    SYNTHETIC_NETWORKS_PATH = os.path.join(_RESULTS, "synthetic")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.NETWORKS_FUNC = staticmethod(_load_networks)


def _load_networks(folder: str) -> list:
    from graphs import load_graphs

    return load_graphs(folder)


def _load_network_csv(folder: str) -> list:
    from graphs import read_graph_csv

    graphs = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith("_edgelist.csv"):
            continue
        positions = os.path.join(
            folder, name.replace("_edgelist.csv", "_positions.csv")
        )
        assert os.path.exists(positions), (
            f"{os.path.join(folder, name)} has no positions file "
            f"beside it ({positions})"
        )
        graphs.append(read_graph_csv(os.path.join(folder, name), positions))
    return graphs
