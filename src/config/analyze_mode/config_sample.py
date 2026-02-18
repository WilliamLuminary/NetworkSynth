# src/config/analyze_mode/config_sample.py
import logging
import os
import pickle
from typing import List, Tuple

from graph.synth_graph import SynthGraph

from ..base_config import BaseConfig

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MEASURE_WEIGHTED = False
    NETWORKS_DATA_PATH = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "results_multi")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.NETWORKS_FUNC = cls._load_networks_dict
        cls._inject_dependencies()

    @staticmethod
    def _load_networks_dict(
        _both_networks_path,
    ) -> Tuple[List[SynthGraph], List[SynthGraph]]:
        original_network, synthetic_networks = None, None
        for entry in os.listdir(_both_networks_path):
            entry_path = os.path.join(_both_networks_path, entry)
            if entry == "synthetic":
                synthetic_networks = _load_network_pkl(entry_path)
            elif entry in ("origin", "original"):
                original_network = _load_network_pkl(entry_path)
        return original_network, synthetic_networks


def _load_network_pkl(folder: str) -> List[SynthGraph]:
    """Load network pkl files with backward compatibility for nx.Graph."""
    import networkx as nx

    for file in os.listdir(folder):
        if file.endswith(".pkl") and "network" in file:
            with open(os.path.join(folder, file), "rb") as f:
                content = pickle.load(f)
                if isinstance(content, nx.Graph):
                    return [SynthGraph.from_networkx(content)]
                elif isinstance(content, SynthGraph):
                    return [content]
                elif isinstance(content, list):
                    return [
                        SynthGraph.from_networkx(g) if isinstance(g, nx.Graph) else g
                        for g in content
                    ]
    return []
