"""
Convert a NetworkX pickle file to CSV (edge list + positions) and NetworKit binary.

Usage:
    python scripts/convert_pkl_network.py <pkl_path> [--output-dir <dir>]
"""

import argparse
import csv
import os
import pickle
import sys

import networkit as nk
import networkx as nx


def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        print(f"Pickle contains {len(data)} graph(s)")
        return data
    if isinstance(data, nx.Graph):
        return [data]
    raise TypeError(f"Unexpected type: {type(data)}")


def save_edgelist_csv(graph: nx.Graph, filepath: str) -> None:
    pos = nx.get_node_attributes(graph, "pos")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_x", "source_y", "target_x", "target_y", "weight"])
        for u, v, d in graph.edges(data=True):
            w.writerow([*pos[u], *pos[v], d.get("weight", 1.0)])
    print(f"  Edgelist CSV: {filepath} ({graph.number_of_edges():,} edges)")


def save_positions_csv(graph: nx.Graph, filepath: str) -> None:
    pos = nx.get_node_attributes(graph, "pos")
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for p in pos.values():
            w.writerow([p[0], p[1]])
    print(f"  Positions CSV: {filepath} ({len(pos):,} nodes)")


def save_networkit(graph: nx.Graph, filepath: str) -> None:
    n = graph.number_of_nodes()
    weighted = any("weight" in d for _, _, d in graph.edges(data=True))
    nk_graph = nk.Graph(n, weighted=weighted)
    node_map = {node: idx for idx, node in enumerate(graph.nodes())}
    for u, v, d in graph.edges(data=True):
        nk_graph.addEdge(
            node_map[u], node_map[v], d.get("weight", 1.0) if weighted else 1.0
        )
    nk.writeGraph(nk_graph, filepath, nk.Format.NetworkitBinary)
    print(
        f"  NetworKit binary: {filepath} ({n:,} nodes, {graph.number_of_edges():,} edges)"
    )


def convert(pkl_path: str, output_dir: str = None) -> None:
    if not os.path.isfile(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)

    graphs = load_pkl(pkl_path)
    if not graphs:
        print("No graphs found.")
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.dirname(pkl_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    for i, graph in enumerate(graphs):
        suffix = f"_n{i}" if len(graphs) > 1 else ""
        prefix = f"{base_name}{suffix}"
        print(
            f"\nGraph {i}: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges"
        )
        save_edgelist_csv(graph, os.path.join(output_dir, f"{prefix}_edgelist.csv"))
        save_positions_csv(graph, os.path.join(output_dir, f"{prefix}_positions.csv"))
        save_networkit(graph, os.path.join(output_dir, f"{prefix}.nkbin"))

    print(f"\nDone. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NetworkX pkl to CSV + NetworKit binary."
    )
    parser.add_argument("pkl_path", help="Path to the NetworkX pickle file")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: same as pkl)"
    )
    args = parser.parse_args()
    convert(args.pkl_path, args.output_dir)


if __name__ == "__main__":
    main()
