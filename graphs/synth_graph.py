from __future__ import annotations

from typing import Iterator, List, Set, Tuple

import networkit as nk
import numpy as np


class SynthGraph:

    __slots__ = ("_graph", "_positions")

    def __init__(self, graph: nk.Graph, positions: np.ndarray):
        self._graph: nk.Graph = graph
        self._positions: np.ndarray = np.asarray(positions, dtype=np.float64)

    @property
    def nk(self) -> nk.Graph:
        return self._graph

    def number_of_nodes(self) -> int:
        return self._graph.numberOfNodes()

    def number_of_edges(self) -> int:
        return self._graph.numberOfEdges()

    def is_weighted(self) -> bool:
        return self._graph.isWeighted()

    def positions(self) -> np.ndarray:
        return self._positions

    def position(self, node: int) -> np.ndarray:
        return self._positions[node]

    def set_position(self, node: int, pos) -> None:
        self._positions[node] = pos

    def degree(self, node: int) -> int:
        return self._graph.degree(node)

    def degrees(self) -> List[Tuple[int, int]]:
        return [(u, self._graph.degree(u)) for u in self._graph.iterNodes()]

    def degree_sequence(self) -> List[int]:
        return [self._graph.degree(u) for u in self._graph.iterNodes()]

    def weighted_degree(self, node: int) -> float:
        return self._graph.weightedDegree(node)

    def weight(self, u: int, v: int) -> float:
        return self._graph.weight(u, v)

    def set_weight(self, u: int, v: int, w: float) -> None:
        assert w > 0, f"edge ({u}, {v}) has non-positive weight {w!r}"
        self._graph.setWeight(u, v, w)

    def has_edge(self, u: int, v: int) -> bool:
        return self._graph.hasEdge(u, v)

    def remove_edge(self, u: int, v: int) -> None:
        self._graph.removeEdge(u, v)

    def nodes(self) -> Iterator[int]:
        return self._graph.iterNodes()

    def neighbors(self, node: int) -> List[int]:
        return list(self._graph.iterNeighbors(node))

    def edges(self) -> Iterator[Tuple[int, int]]:
        return self._graph.iterEdges()

    def edges_with_weights(self) -> Iterator[Tuple[int, int, float]]:
        return self._graph.iterEdgesWeights()

    def is_connected(self) -> bool:
        cc = nk.components.ConnectedComponents(self._graph)
        cc.run()
        return cc.numberOfComponents() == 1

    def largest_connected_component(self) -> SynthGraph:
        n = self._graph.numberOfNodes()
        if n <= 1:
            return self.copy()
        cc = nk.components.ConnectedComponents(self._graph)
        cc.run()
        if cc.numberOfComponents() == 1:
            return self.copy()
        components = cc.getComponents()
        largest = max(components, key=len)
        return self.subgraph(set(largest))

    def subgraph(self, node_set: Set[int]) -> SynthGraph:
        node_list = sorted(node_set)
        node_map = {old: new for new, old in enumerate(node_list)}
        n = len(node_list)
        weighted = self._graph.isWeighted()
        new_graph = nk.Graph(n, weighted=weighted)
        for u, v in self._graph.iterEdges():
            if u in node_map and v in node_map:
                w = self._graph.weight(u, v) if weighted else 1.0
                new_graph.addEdge(node_map[u], node_map[v], w)
        new_positions = self._positions[node_list]
        return SynthGraph(new_graph, new_positions)

    def copy(self) -> SynthGraph:
        return SynthGraph(
            nk.Graph(self._graph, weighted=self._graph.isWeighted()),
            self._positions.copy(),
        )

    def make_weighted(self) -> None:
        if self._graph.isWeighted():
            return
        n = self._graph.numberOfNodes()
        new_graph = nk.Graph(n, weighted=True)
        for u, v in self._graph.iterEdges():
            new_graph.addEdge(u, v, 1.0)
        self._graph = new_graph

    def make_unweighted(self) -> None:
        if not self._graph.isWeighted():
            return
        n = self._graph.numberOfNodes()
        new_graph = nk.Graph(n, weighted=False)
        for u, v in self._graph.iterEdges():
            new_graph.addEdge(u, v)
        self._graph = new_graph

    @classmethod
    def from_networkx(cls, G) -> SynthGraph:
        import networkx as nx

        G = nx.convert_node_labels_to_integers(G)
        weighted = any("weight" in d for _, _, d in G.edges(data=True))
        nk_graph = nk.nxadapter.nx2nk(G, weightAttr="weight" if weighted else None)
        n = G.number_of_nodes()
        positions = np.zeros((n, 2), dtype=np.float64)
        pos_dict = nx.get_node_attributes(G, "pos")
        for i in range(n):
            if i in pos_dict:
                positions[i] = pos_dict[i]
        return cls(nk_graph, positions)

    @classmethod
    def from_sparse_matrix(cls, positions: np.ndarray, adjacency_matrix) -> SynthGraph:
        import scipy.sparse as sp

        coo = (
            adjacency_matrix.tocoo()
            if sp.issparse(adjacency_matrix)
            else sp.coo_matrix(adjacency_matrix)
        )
        n = coo.shape[0]
        nk_graph = nk.Graph(n, weighted=True)
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i != j and not nk_graph.hasEdge(int(i), int(j)):
                assert v > 0, f"edge ({i}, {j}) has non-positive weight {v!r}"
                nk_graph.addEdge(int(i), int(j), float(v))
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 2)
        return cls(nk_graph, pos.copy())

    @classmethod
    def from_graph_nodes(cls, nodes: set, edges: set) -> SynthGraph:
        position_map = {node.position: node for node in nodes}
        sorted_nodes = sorted(nodes, key=lambda x: x.id)
        n = len(sorted_nodes)
        nk_graph = nk.Graph(n, weighted=False)
        positions = np.zeros((n, 2), dtype=np.float64)
        id_map = {}
        for new_id, node in enumerate(sorted_nodes):
            id_map[node.id] = new_id
            positions[new_id] = node.position
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
                if not nk_graph.hasEdge(uid, vid):
                    nk_graph.addEdge(uid, vid)
        return cls(nk_graph, positions)

    @classmethod
    def from_edge_list(cls, positions: np.ndarray, edge_list: np.ndarray) -> SynthGraph:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 2)
        n = len(pos)
        nk_graph = nk.Graph(n, weighted=False)
        for edge in edge_list:
            u, v = int(edge[0]), int(edge[1])
            if not nk_graph.hasEdge(u, v):
                nk_graph.addEdge(u, v)
        return cls(nk_graph, pos.copy())

    def __repr__(self) -> str:
        return (
            f"SynthGraph(nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()}, "
            f"weighted={self.is_weighted()})"
        )
