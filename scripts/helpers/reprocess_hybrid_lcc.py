# scripts/reprocess_hybrid_lcc.py
"""
Post-process the existing hybrid 50x50 LCC output.

Loads the LCC graph from .nkbin + .npy, assigns edge weights via the
original-network Mapper, and re-saves as consistent CSV files
(edgelist with weights, positions as CSV instead of .npy).

Usage (from project root):
    python scripts/reprocess_hybrid_lcc.py
"""
import csv
import logging
import os
import sys

import networkit as nk
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SYNTH_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "output",
    "SampleConfig_results_20260226_005320",
    "sample_1",
    "synthetic",
)

NKBIN_PATH = os.path.join(SYNTH_DIR, "hybrid_50x50_lcc.nkbin")
NPY_PATH = os.path.join(SYNTH_DIR, "hybrid_50x50_lcc_positions.npy")

OUT_EDGELIST = os.path.join(SYNTH_DIR, "hybrid_50x50_lcc_edgelist.csv")
OUT_POSITIONS = os.path.join(SYNTH_DIR, "hybrid_50x50_lcc_positions.csv")

from configs.hybrid_mode import HybridConfig

HybridConfig.initialize()
HybridConfig.disable_saving("reprocessing only")

from graphs.synth_graph import SynthGraph
from handlers import RunAgent, create_run_paths

logger.info(f"Loading graph from {NKBIN_PATH}")
nk_graph = nk.readGraph(NKBIN_PATH, nk.Format.NetworkitBinary)
logger.info(f"  nodes={nk_graph.numberOfNodes():,}  edges={nk_graph.numberOfEdges():,}")

logger.info(f"Loading positions from {NPY_PATH}")
positions = np.load(NPY_PATH)
logger.info(f"  shape={positions.shape}")

assert (
    positions.shape[0] == nk_graph.numberOfNodes()
), f"Position count {positions.shape[0]} != node count {nk_graph.numberOfNodes()}"

graph = SynthGraph(nk_graph, positions)

dataset_id = HybridConfig.get_datasets()[0]
run_paths = create_run_paths(HybridConfig)
data_agent = RunAgent(HybridConfig, run_paths=run_paths, dataset_id=dataset_id)
data_agent.prepare_data()
mapper = data_agent.mapper

logger.info("Assigning edge weights via Mapper (length → weight mapping)")
mapper.assign_weights(graph)

logger.info(f"Writing edgelist → {OUT_EDGELIST}")
with open(OUT_EDGELIST, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_index", "target_index", "edge_weight"])
    for u, v, w in graph.edges_with_weights():
        writer.writerow([u, v, w])

logger.info(f"Writing positions → {OUT_POSITIONS}")
with open(OUT_POSITIONS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    pos = graph.positions()
    for row in pos:
        writer.writerow([row[0], row[1]])

logger.info("Done")
