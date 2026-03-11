# scripts/run_scaling_100x100.py
"""
Production run: generate a 100x100 scaled network via multi-root BFS.

This uses the "scaling" pipeline (``pipelines.scaling``) approach —
a single-process synchronized BFS with 100x100 = 10,000 root nodes
sharing the same spatial grids.  Components merge naturally as
branches from different roots encounter each other.

Logs are written to ``data/output/logs/scaling_100x100.log`` for
real-time monitoring via ``tail -f``.

Usage (from project root):
    python scripts/run_scaling_100x100.py
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "scaling_100x100.log")

file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)
logger = logging.getLogger(__name__)

from configs.scaling_mode import ScalingConfig

ScalingConfig.initialize()

import networkit as nk

from configs import BaseConfig, DataType
from graphs import GraphGenerator
from handlers import RunAgent, Saver
from utils import trim_graph

Saver.initialize()
data_agent = RunAgent(dataset_id=BaseConfig.get_datasets()[0])
data_agent.prepare_data()
attributes = data_agent.attributes

logger.info(f"average_length: {attributes.average_length:.2f}")
logger.info(f"average_degree: {attributes.average_degree:.2f}")

t0 = time.time()
generator = GraphGenerator(attributes)
scaled_graph = generator.generate_scaled_network(
    scale_rows=100,
    scale_cols=100,
    max_rounds=500,
    root_spacing_factor=1.0,
)
elapsed = time.time() - t0
logger.info(f"Generation took {elapsed:.1f}s ({elapsed/60:.1f}min)")

cc = nk.components.ConnectedComponents(scaled_graph.nk)
cc.run()
n_comp = cc.numberOfComponents()
sizes = sorted([len(c) for c in cc.getComponents()], reverse=True)
logger.info(f"Nodes: {scaled_graph.number_of_nodes():,}")
logger.info(f"Edges: {scaled_graph.number_of_edges():,}")
logger.info(f"Connected components: {n_comp}")
logger.info(f"Top 5 sizes: {sizes[:5]}")

scaled_graph = trim_graph(scaled_graph, attributes.average_degree)
data_agent.mapper.assign_weights(scaled_graph)
data_agent.add_synthetic_graph(scaled_graph)
prefix = "scaled_100x100_"
data_agent.save(DataType.SYNTHETIC_EDGELIST, prefix, arg=scaled_graph)
data_agent.save(DataType.SYNTHETIC_POSITIONS, prefix, arg=scaled_graph)
data_agent.save(DataType.SYNTHETIC_NETWORK_NKI, prefix, arg=scaled_graph)

logger.info("DONE")
