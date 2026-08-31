import csv
import os

import networkit as nk
import numpy as np

BASE = os.path.join("data")
PREFIX = "hybrid_50x50_lcc"


def from_csvs():
    positions = np.loadtxt(
        os.path.join(BASE, f"{PREFIX}_positions.csv"), delimiter=",", skiprows=1
    )

    edges, weights = [], []
    with open(os.path.join(BASE, f"{PREFIX}_edgelist.csv")) as f:
        reader = csv.reader(f)
        header = next(reader)
        has_weight = len(header) >= 3
        for row in reader:
            edges.append((int(row[0]), int(row[1])))
            if has_weight:
                weights.append(float(row[2]))

    g = nk.Graph(len(positions), weighted=has_weight)
    for i, (u, v) in enumerate(edges):
        g.addEdge(u, v, weights[i] if has_weight else 1.0)

    print(
        f"[CSV]   nodes={g.numberOfNodes():,}  edges={g.numberOfEdges():,}"
        f"  weighted={g.isWeighted()}"
    )
    return g, positions


def from_graphml():
    from graphs.graph_loader import load_graphs

    g = load_graphs(os.path.join(BASE, f"{PREFIX}.graphml.gz"))[0]
    print(
        f"[graphml] nodes={g.number_of_nodes():,}  edges={g.number_of_edges():,}"
        f"  weighted={g.is_weighted()}"
    )
    return g, g.positions()


if __name__ == "__main__":
    print("--- CSV pair ---")
    g1, p1 = from_csvs()

    print("--- GraphML ---")
    g2, p2 = from_graphml()
