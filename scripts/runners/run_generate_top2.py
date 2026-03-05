# scripts/run_generate_top2.py
"""
Generate networks for the top-2 hyperparameter settings from a sweep.

Takes the two best (CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR) pairs
found via ``pipelines.sweep`` and generates 2 synthetic networks (with
plots) for each, using GenConfigTmp.

Usage (from project root):
    python scripts/run_generate_top2.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging

from configs import BaseConfig, DataType
from configs.generate_mode import GenConfigTmp as GenConfig
from handlers import RunAgent, Saver
from pipelines.generate import generate_with_multiprocessing

GenConfig.initialize()

SETTINGS = [
    {"nf": 1.3, "ef": 1.9, "label": "nf1.3_ef1.9"},
    {"nf": 0.4, "ef": 1.9, "label": "nf0.4_ef1.9"},
]

BaseConfig.SYNTHETIC_NETWORK_NUMBER = 2
BaseConfig.SYNTHETIC_GRAPH_NUMBER = 2
BaseConfig.MAX_ATTEMPTS = 10
BaseConfig.ERROR_TOLERANCE = 0.15

logger = logging.getLogger(__name__)


def main():
    Saver.initialize()
    dataset_id = BaseConfig.get_datasets()[0]

    for setting in SETTINGS:
        nf, ef, label = setting["nf"], setting["ef"], setting["label"]
        BaseConfig.set_node_factor(nf)
        BaseConfig.set_edge_factor(ef)

        logger.info(f"=== Generating with {label} (nf={nf}, ef={ef}) ===")

        data_agent = RunAgent(dataset_id=dataset_id)
        data_agent.prepare_data()
        data_agent.save(DataType.ORIGINAL_IMAGE)
        data_agent.save(DataType.ORIGINAL_NETWORK)
        data_agent.save(DataType.ORIGINAL_GRAPH)

        generate_with_multiprocessing(data_agent)

        logger.info(f"=== Done with {label} ===\n")


if __name__ == "__main__":
    main()
