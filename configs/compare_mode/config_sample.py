from __future__ import annotations

import logging
import os
import pickle
from typing import List

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MODE = "compare"

    MEASURE_WEIGHTED = False

    #: One dataset per run: analysis compares the two sets below, and which two
    #: they are is the point of the mode, not something to discover.
    DATASETS = [DatasetId("analysis")]

    _RESULTS = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "latest_result", "sample_1")
    ORIGINAL_NETWORKS_PATH = os.path.join(_RESULTS, "original")
    SYNTHETIC_NETWORKS_PATH = os.path.join(_RESULTS, "synthetic")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.NETWORKS_FUNC = staticmethod(_load_networks)


def _load_networks(folder: str) -> list:
    """Every network in *folder*, in whichever format it was written.

    Pickles first, because one holds a whole batch, then the CSV pairs — the
    original network is only ever written as CSV, so a results directory cannot
    be read at all without both.  Either format goes through the same readers
    the generate mode uses; nothing here parses a graph itself.
    """
    pickled = _load_network_pkl(folder)
    if pickled:
        return pickled
    return _load_network_csv(folder)


def _load_network_csv(folder: str) -> list:
    from graphs import read_graph_csv

    graphs = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith("_edgelist.csv"):
            continue
        positions = os.path.join(
            folder, name.replace("_edgelist.csv", "_positions.csv")
        )
        # Named and stopped rather than skipped: half a results directory
        # analysed as if it were whole is a wrong answer, not a smaller one.
        assert os.path.exists(positions), (
            f"{os.path.join(folder, name)} has no positions file "
            f"beside it ({positions})"
        )
        graphs.append(read_graph_csv(os.path.join(folder, name), positions))
    return graphs


def _load_network_pkl(folder: str) -> list:
    """Load network pkl files with backward compatibility for nx.Graph.

    If the pickle contains legacy ``nx.Graph`` objects, *networkx*
    must be installed so ``pickle.load`` can deserialize them.
    """
    from graphs.synth_graph import SynthGraph

    try:
        import networkx as nx

        _has_nx = True
    except ImportError:
        nx = None  # type: ignore[assignment]
        _has_nx = False

    for file in os.listdir(folder):
        if file.endswith(".pkl") and "network" in file:
            try:
                with open(os.path.join(folder, file), "rb") as f:
                    content = pickle.load(f)
            except ModuleNotFoundError as exc:
                if "networkx" in str(exc):
                    logger.error(
                        "Legacy nx.Graph pickle but "
                        "networkx not installed. "
                        "pip install networkx"
                    )
                raise

            if isinstance(content, SynthGraph):
                return [content]
            if _has_nx and isinstance(content, nx.Graph):
                return [SynthGraph.from_networkx(content)]
            if isinstance(content, list):
                result: List[SynthGraph] = []
                for g in content:
                    if isinstance(g, SynthGraph):
                        result.append(g)
                    elif _has_nx and isinstance(g, nx.Graph):
                        result.append(SynthGraph.from_networkx(g))
                return result
    return []
