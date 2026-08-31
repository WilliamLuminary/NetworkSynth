from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from graphs.synth_graph import SynthGraph


def compute_network_metrics(graph: SynthGraph) -> dict:
    from scipy.spatial.distance import euclidean

    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    node_count = n
    avg_degree = 2.0 * e / n if n > 0 else 0.0

    scores = graph.local_clustering()
    avg_clustering = sum(scores) / n if n > 0 else 0.0

    positions = graph.positions()
    total_length = 0.0
    for u, v in graph.edges():
        total_length += euclidean(positions[u], positions[v])
    avg_length = total_length / e if e > 0 else 0.0

    all_angle_diffs = []
    for node in graph.nodes():
        nbrs = graph.neighbors(node)
        if len(nbrs) < 2:
            continue
        node_pos = positions[node]
        angles = [
            np.arctan2(
                positions[nbr][1] - node_pos[1],
                positions[nbr][0] - node_pos[0],
            )
            * 180.0
            / np.pi
            for nbr in nbrs
        ]
        sorted_angles = np.sort(angles)
        diffs = np.diff(sorted_angles)
        diffs = np.append(diffs, 360.0 + sorted_angles[0] - sorted_angles[-1])
        all_angle_diffs.extend(diffs)
    avg_angle = float(np.mean(all_angle_diffs)) if all_angle_diffs else 0.0

    return {
        "node_count": node_count,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "avg_length": avg_length,
        "avg_angle": avg_angle,
    }


def metric_distance(metrics: dict, ref_metrics: dict) -> float:
    total = 0.0
    for key in ref_metrics:
        ref = ref_metrics[key]
        syn = metrics[key]
        total += abs(syn - ref) / abs(ref) if ref != 0 else abs(syn)
    return total / len(ref_metrics)
