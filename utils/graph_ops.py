from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from graphs.synth_graph import SynthGraph


def calculate_frame(
    graph: SynthGraph = None,
    center_position: Union[tuple, list, ndarray] = None,
    *,
    frame_range: Tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if graph is None:
        if not center_position:
            raise ValueError(
                "Either synthetic_graph or center_position must be provided."
            )
        _center_x, _center_y = center_position
    else:
        if center_position:
            raise ValueError(
                "Only synthetic_graph or center_position must be provided."
            )
        _positions = graph.positions()
        _center_x, _center_y = (
            _positions[:, 0].mean(),
            _positions[:, 1].mean(),
        )

    _half_range = (frame_range[0] / 2, frame_range[1] / 2)
    _width, _height = _half_range

    frame = (
        (
            round(_center_x - _width, 2),
            round(_center_x + _width, 2),
        ),
        (
            round(_center_y - _height, 2),
            round(_center_y + _height, 2),
        ),
    )
    return frame


def build_graph(*args, arg_type: str = "adjacency_matrix") -> SynthGraph:
    if arg_type == "adjacency_matrix":
        return _from_adjacency_matrix(*args)
    elif arg_type == "graph_node":
        return _from_graph_node(*args)
    elif arg_type == "edge_list":
        return _from_edge_list(*args)
    else:
        raise TypeError(
            f"Unsupported Type: {arg_type} from function {build_graph.__name__}."
        )


def _from_adjacency_matrix(positions: np.ndarray, adjacency_matrix) -> SynthGraph:
    from graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_sparse_matrix(positions, adjacency_matrix)
    return graph.largest_connected_component()


def _from_graph_node(nodes: set, edges: set) -> SynthGraph:
    from graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_graph_nodes(nodes, edges)
    return graph.largest_connected_component()


def _from_edge_list(positions: np.ndarray, edge_list: np.ndarray) -> SynthGraph:
    from graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_edge_list(positions, edge_list)
    return graph.largest_connected_component()


def largest_connected_component(graph: SynthGraph) -> SynthGraph:
    return graph.largest_connected_component()


def trim_graph(graph: SynthGraph, tar_avg_deg: float) -> SynthGraph:
    """Trim edges from high-degree nodes until average degree <= 1.1 * target.

    Uses vectorised NumPy to score all edges and ``argpartition`` to
    select which to keep in O(E), then rebuilds the graph and extracts
    the LCC.  Repeats if LCC shifts the average degree above target.
    """
    import networkit as nk

    logger = logging.getLogger(__name__)
    target = 1.1 * tar_avg_deg

    round_num = 0
    while True:
        n = graph.number_of_nodes()
        if n == 0:
            return graph
        m = graph.number_of_edges()
        current_avg = 2 * m / n
        if current_avg <= target:
            break

        target_edges = int(target * n / 2)
        logger.debug(
            f"trim_graph round {round_num}: avg_deg={current_avg:.2f} → "
            f"target≤{target:.2f}, keeping {target_edges:,} of "
            f"{m:,} edges ({n:,} nodes)"
        )

        src = np.empty(m, dtype=np.int64)
        dst = np.empty(m, dtype=np.int64)
        for i, (u, v) in enumerate(graph.edges()):
            src[i] = u
            dst[i] = v

        degrees = np.array([graph.degree(u) for u in range(n)], dtype=np.int32)
        edge_scores = np.maximum(degrees[src], degrees[dst])
        keep_idx = np.argpartition(edge_scores, target_edges)[:target_edges]

        kept_src = src[keep_idx]
        kept_dst = dst[keep_idx]
        weighted = graph.is_weighted()
        new_nk = nk.Graph(n, weighted=weighted)
        if weighted:
            for i in range(len(kept_src)):
                new_nk.addEdge(
                    int(kept_src[i]),
                    int(kept_dst[i]),
                    graph.weight(int(kept_src[i]), int(kept_dst[i])),
                )
        else:
            for i in range(len(kept_src)):
                new_nk.addEdge(int(kept_src[i]), int(kept_dst[i]))

        from graphs.synth_graph import SynthGraph

        graph = SynthGraph(new_nk, graph.positions())
        logger.debug(
            f"trim_graph round {round_num}: rebuilt with "
            f"{graph.number_of_edges():,} edges, extracting LCC"
        )
        graph = graph.largest_connected_component()
        round_num += 1

    return graph


def transpose_positions(graph) -> None:
    """Swap x and y for every node, in place.

    Some inputs record positions row-major (row, column) where the rest of the
    toolkit expects (x, y).
    """
    positions = graph.positions()
    positions[:, [0, 1]] = positions[:, [1, 0]]
