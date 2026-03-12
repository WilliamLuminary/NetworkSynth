# src/configs/analyze_mode/config_sample.py
from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING, List, Tuple

from ..base_config import BaseConfig

if TYPE_CHECKING:
    from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MEASURE_WEIGHTED = False
    NETWORKS_DATA_PATH = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "results_multi")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.NETWORKS_FUNC = cls._load_networks_dict
        from ..enums import DataType
        from ..file_definitions import SaveSpec, save_pickle, save_svg

        cls.SAVE_SPECS = {
            DataType.ANALYSIS_DATA: SaveSpec(
                "", "analysis_data", save_pickle, use_timestamp=False
            ),
            DataType.ANALYSIS_FIGURE: SaveSpec("", "analysis_figure", save_svg),
        }
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
