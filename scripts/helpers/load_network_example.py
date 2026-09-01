# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BASE = os.path.join("data")
PREFIX = "hybrid_50x50_lcc"


def from_csvs():
    from graphs.graph_loader import load_graphs

    g = load_graphs(os.path.join(BASE, f"{PREFIX}_edgelist.csv"))[0]
    print(
        f"[CSV]   nodes={g.number_of_nodes():,}  edges={g.number_of_edges():,}"
        f"  weighted={g.is_weighted()}"
    )
    return g, g.positions()


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
