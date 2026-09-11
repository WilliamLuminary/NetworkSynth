# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Iterator, List, Set, Tuple

import igraph as ig
import numpy as np

_WEIGHT = "weight"


class SynthGraph:

    __slots__ = ("_graph", "_positions")

    def __init__(self, graph: ig.Graph, positions: np.ndarray):
        self._graph: ig.Graph = graph
        self._positions: np.ndarray = np.asarray(positions, dtype=np.float64)

    @property
    def igraph(self) -> ig.Graph:
        return self._graph

    def number_of_nodes(self) -> int:
        return self._graph.vcount()

    def number_of_edges(self) -> int:
        return self._graph.ecount()

    def is_weighted(self) -> bool:
        return _WEIGHT in self._graph.es.attributes()

    def positions(self) -> np.ndarray:
        return self._positions

    def position(self, node: int) -> np.ndarray:
        return self._positions[node]

    def set_position(self, node: int, pos) -> None:
        self._positions[node] = pos

    def degree(self, node: int) -> int:
        return self._graph.degree(node)

    def degrees(self) -> List[Tuple[int, int]]:
        return list(enumerate(self._graph.degree()))

    def degree_sequence(self) -> List[int]:
        return self._graph.degree()

    def weighted_degree(self, node: int) -> float:
        if not self.is_weighted():
            return float(self._graph.degree(node))
        return float(self._graph.strength(node, weights=_WEIGHT))

    def weight(self, u: int, v: int) -> float:
        """The edge's weight, or 0.0 when there is no such edge."""
        edge_id = self._graph.get_eid(u, v, error=False)
        if edge_id == -1:
            return 0.0
        if not self.is_weighted():
            return 1.0
        return float(self._graph.es[edge_id][_WEIGHT])

    def set_weight(self, u: int, v: int, w: float) -> None:
        assert w > 0, f"edge ({u}, {v}) has non-positive weight {w!r}"
        self._graph.es[self._graph.get_eid(u, v)][_WEIGHT] = w

    def has_edge(self, u: int, v: int) -> bool:
        return self._graph.get_eid(u, v, error=False) != -1

    def remove_edge(self, u: int, v: int) -> None:
        self._graph.delete_edges([(u, v)])

    def nodes(self) -> Iterator[int]:
        return iter(range(self._graph.vcount()))

    def neighbors(self, node: int) -> List[int]:
        return self._graph.neighbors(node)

    def edges(self) -> Iterator[Tuple[int, int]]:
        return iter(self._graph.get_edgelist())

    def edges_with_weights(self) -> Iterator[Tuple[int, int, float]]:
        pairs = self._graph.get_edgelist()
        weights = (
            self._graph.es[_WEIGHT]
            if self.is_weighted()
            else [1.0] * self._graph.ecount()
        )
        return iter([(u, v, float(w)) for (u, v), w in zip(pairs, weights)])

    def is_connected(self) -> bool:
        return self._graph.is_connected()

    def component_sizes(self) -> List[int]:
        return [len(component) for component in self._graph.connected_components()]

    def local_clustering(self) -> List[float]:
        return self._graph.transitivity_local_undirected(mode="zero")

    def largest_connected_component(self) -> SynthGraph:
        if self._graph.vcount() <= 1 or self._graph.is_connected():
            return self.copy()
        largest = max(self._graph.connected_components(), key=len)
        return self.subgraph(set(largest))

    def subgraph(self, node_set: Set[int]) -> SynthGraph:
        node_list = sorted(node_set)
        return SynthGraph(
            self._graph.induced_subgraph(node_list),
            self._positions[node_list],
        )

    def copy(self) -> SynthGraph:
        return SynthGraph(self._graph.copy(), self._positions.copy())

    def make_weighted(self) -> None:
        if self.is_weighted():
            return
        self._graph.es[_WEIGHT] = [1.0] * self._graph.ecount()

    def make_unweighted(self) -> None:
        if not self.is_weighted():
            return
        del self._graph.es[_WEIGHT]

    @classmethod
    def from_sparse_matrix(cls, positions: np.ndarray, adjacency_matrix) -> SynthGraph:
        import scipy.sparse as sp

        coo = (
            adjacency_matrix.tocoo()
            if sp.issparse(adjacency_matrix)
            else sp.coo_matrix(adjacency_matrix)
        )
        n = coo.shape[0]
        seen: Set[Tuple[int, int]] = set()
        pairs: List[Tuple[int, int]] = []
        weights: List[float] = []
        for i, j, v in zip(coo.row, coo.col, coo.data):
            u, w = int(i), int(j)
            if u == w or (min(u, w), max(u, w)) in seen:
                continue
            assert v > 0, f"edge ({i}, {j}) has non-positive weight {v!r}"
            seen.add((min(u, w), max(u, w)))
            pairs.append((u, w))
            weights.append(float(v))
        graph = ig.Graph(n=n, edges=pairs)
        graph.es[_WEIGHT] = weights
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 2)
        return cls(graph, pos.copy())

    @classmethod
    def from_graph_nodes(cls, nodes: set, edges: set) -> SynthGraph:
        sorted_nodes = sorted(nodes, key=lambda x: x.id)
        position_map = {node.position: node for node in sorted_nodes}
        n = len(sorted_nodes)
        positions = np.zeros((n, 2), dtype=np.float64)
        id_map = {}
        for new_id, node in enumerate(sorted_nodes):
            id_map[node.id] = new_id
            positions[new_id] = node.position
        seen: Set[Tuple[int, int]] = set()
        pairs: List[Tuple[int, int]] = []
        for edge in edges:
            u_node = position_map.get(edge[0])
            v_node = position_map.get(edge[1])
            if (
                u_node is not None
                and v_node is not None
                and u_node.id in id_map
                and v_node.id in id_map
            ):
                uid, vid = id_map[u_node.id], id_map[v_node.id]
                key = (min(uid, vid), max(uid, vid))
                if uid != vid and key not in seen:
                    seen.add(key)
                    pairs.append((uid, vid))
        return cls(ig.Graph(n=n, edges=pairs), positions)

    @classmethod
    def from_edge_list(cls, positions: np.ndarray, edge_list: np.ndarray) -> SynthGraph:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 2)
        n = len(pos)
        seen: Set[Tuple[int, int]] = set()
        pairs: List[Tuple[int, int]] = []
        for edge in edge_list:
            u, v = int(edge[0]), int(edge[1])
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                pairs.append((u, v))
        return cls(ig.Graph(n=n, edges=pairs), pos.copy())

    def __repr__(self) -> str:
        return (
            f"SynthGraph(nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()}, "
            f"weighted={self.is_weighted()})"
        )
