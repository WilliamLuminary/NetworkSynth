# scripts/load_network_example.py
"""
Minimal examples showing how to reconstruct a SynthGraph from saved files.

Method 1: from edgelist.csv + positions.csv  (portable, human-readable)
Method 2: from .nkbin + positions.csv        (fast binary load)
"""
import csv

import networkit as nk
import numpy as np


def from_csvs(edgelist_path: str, positions_path: str):
    """Load from edgelist CSV + positions CSV."""
    positions = np.loadtxt(positions_path, delimiter=",", skiprows=1)

    edges = []
    weights = []
    with open(edgelist_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        has_weight = len(header) >= 3
        for row in reader:
            edges.append((int(row[0]), int(row[1])))
            if has_weight:
                weights.append(float(row[2]))

    n = len(positions)
    g = nk.Graph(n, weighted=has_weight)
    for i, (u, v) in enumerate(edges):
        w = weights[i] if has_weight else 1.0
        g.addEdge(u, v, w)

    print(
        f"[CSV]   nodes={g.numberOfNodes():,}  edges={g.numberOfEdges():,}  weighted={g.isWeighted()}"
    )
    return g, positions


def from_nkbin(nkbin_path: str, positions_path: str):
    """Load from .nkbin + positions CSV."""
    g = nk.readGraph(nkbin_path, nk.Format.NetworkitBinary)
    positions = np.loadtxt(positions_path, delimiter=",", skiprows=1)

    assert (
        positions.shape[0] == g.numberOfNodes()
    ), f"Mismatch: {positions.shape[0]} positions vs {g.numberOfNodes()} nodes"

    print(
        f"[nkbin] nodes={g.numberOfNodes():,}  edges={g.numberOfEdges():,}  weighted={g.isWeighted()}"
    )
    return g, positions


if __name__ == "__main__":
    import os

    base = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "output",
        "SampleConfig_results_20260226_005320",
        "sample_1",
        "synthetic",
    )
    prefix = "hybrid_50x50_lcc"

    g1, p1 = from_csvs(
        os.path.join(base, f"{prefix}_edgelist.csv"),
        os.path.join(base, f"{prefix}_positions.csv"),
    )
    g2, p2 = from_nkbin(
        os.path.join(base, f"{prefix}.nkbin"),
        os.path.join(base, f"{prefix}_positions.csv"),
    )
