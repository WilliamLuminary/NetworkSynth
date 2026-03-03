# src/graphs/mosaic_stitcher.py
"""
Mosaic stitcher: combines multiple tile networks into one large network
by merging close nodes at tile boundaries.

The core merging logic mirrors GraphNode's node-merging behavior:
  - Nodes from *different* tiles that are within ``merge_threshold``
    distance are merged into a single node.
  - Spatial hashing (grid cell size = merge_threshold) gives O(1)
    average-case neighbour lookups, identical to GraphNode._spatial_hash.
  - A Union-Find (disjoint-set) structure handles transitive merges
    so that chains of close nodes collapse correctly.

Typical usage:
    merge_thr = avg_length * CLOSED_NODES_FACTOR
    stitcher  = MosaicStitcher(merge_threshold=merge_thr)
    big_graph = stitcher.stitch(tile_graphs)
"""

import logging
from collections import defaultdict
from typing import Dict, Set, Tuple

import networkit as nk
import numpy as np

from .synth_graph import SynthGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set) helper
# ---------------------------------------------------------------------------
class _UnionFind:
    """Weighted quick-union with path compression."""

    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        rank_a = self.rank.get(ra, 0)
        rank_b = self.rank.get(rb, 0)
        if rank_a < rank_b:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if rank_a == rank_b:
            self.rank[ra] = rank_a + 1


# ---------------------------------------------------------------------------
# Stitcher
# ---------------------------------------------------------------------------
class MosaicStitcher:
    """Stitch a grid of tile networks into a single large network."""

    def __init__(self, merge_threshold: float):
        self.merge_threshold = merge_threshold
        self.grid_size = merge_threshold

    def stitch(self, tile_graphs: Dict[Tuple[int, int], SynthGraph]) -> SynthGraph:
        """
        Stitch all tile graphs into one large graph.

        Parameters
        ----------
        tile_graphs : dict[(row, col), SynthGraph]

        Returns
        -------
        SynthGraph
            The stitched network (largest connected component, re-indexed).
        """
        logger.info(f"Stitching {len(tile_graphs)} tiles …")

        combined, node_tile_ids = self._combine_tiles(tile_graphs)
        logger.info(
            f"Combined graph: {combined.number_of_nodes():,} nodes, "
            f"{combined.number_of_edges():,} edges"
        )

        merged = self._merge_close_nodes(combined, node_tile_ids)
        logger.info(
            f"After merge:    {merged.number_of_nodes():,} nodes, "
            f"{merged.number_of_edges():,} edges"
        )

        result = merged.largest_connected_component()
        logger.info(
            f"Final network:  {result.number_of_nodes():,} nodes, "
            f"{result.number_of_edges():,} edges"
        )
        return result

    @staticmethod
    def _combine_tiles(
        tile_graphs: Dict[Tuple[int, int], SynthGraph],
    ) -> Tuple[SynthGraph, np.ndarray]:
        """
        Merge every tile graph into *one* SynthGraph, giving each node a
        globally unique integer id.

        Returns
        -------
        combined : SynthGraph
        node_tile_ids : np.ndarray of shape (N,)
            Integer tile index for each global node (used to identify
            cross-tile pairs during merging).
        """
        total_nodes = sum(g.number_of_nodes() for g in tile_graphs.values())
        all_positions = np.zeros((total_nodes, 2), dtype=np.float64)
        node_tile_ids = np.zeros(total_nodes, dtype=np.int32)
        combined_graph = nk.Graph(total_nodes, weighted=False)

        offset = 0
        for tile_idx, ((row, col), tile) in enumerate(tile_graphs.items()):
            n = tile.number_of_nodes()
            all_positions[offset : offset + n] = tile.positions()
            node_tile_ids[offset : offset + n] = tile_idx

            for u, v in tile.edges():
                combined_graph.addEdge(offset + u, offset + v)

            offset += n

        return SynthGraph(combined_graph, all_positions), node_tile_ids

    def _merge_close_nodes(
        self,
        graph: SynthGraph,
        node_tile_ids: np.ndarray,
    ) -> SynthGraph:
        """
        Find cross-tile node pairs closer than ``merge_threshold`` and
        merge them with a Union-Find structure, then contract the graph.
        """
        positions = graph.positions()

        # --- spatial hash ---
        spatial_hash: Dict[Tuple[int, int], list] = defaultdict(list)
        for node in graph.nodes():
            key = self._spatial_hash(positions[node])
            spatial_hash[key].append(node)

        # --- discover merge candidates ---
        merge_pairs = []
        seen: Set[Tuple[int, int]] = set()

        for key, cell_nodes in spatial_hash.items():
            neighbor_keys = [
                (key[0] + dx, key[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)
            ]
            neighbor_nodes = []
            for nk_key in neighbor_keys:
                neighbor_nodes.extend(spatial_hash.get(nk_key, []))

            for node_a in cell_nodes:
                tile_a = node_tile_ids[node_a]
                pos_a = positions[node_a]

                for node_b in neighbor_nodes:
                    if node_a >= node_b:
                        continue
                    if node_tile_ids[node_b] == tile_a:
                        continue

                    pair_key = (node_a, node_b)
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    pos_b = positions[node_b]
                    dist = float(np.linalg.norm(pos_a - pos_b))
                    if dist < self.merge_threshold:
                        merge_pairs.append((dist, node_a, node_b))

        logger.info(f"Found {len(merge_pairs):,} cross-tile node pairs to merge")

        merge_pairs.sort()

        # --- union-find ---
        uf = _UnionFind()
        for _, a, b in merge_pairs:
            uf.union(a, b)

        # --- contract graph: map old nodes to canonical representatives ---
        canonical = {node: uf.find(node) for node in graph.nodes()}
        kept_nodes = sorted(set(canonical.values()))
        new_id_map = {old: new for new, old in enumerate(kept_nodes)}

        n_new = len(kept_nodes)
        new_graph = nk.Graph(n_new, weighted=False)
        new_positions = np.zeros((n_new, 2), dtype=np.float64)

        for old_id, new_id in new_id_map.items():
            new_positions[new_id] = positions[old_id]

        edge_set: Set[Tuple[int, int]] = set()
        for u, v in graph.edges():
            cu = new_id_map[canonical[u]]
            cv = new_id_map[canonical[v]]
            if cu != cv:
                edge_key = (min(cu, cv), max(cu, cv))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    new_graph.addEdge(cu, cv)

        actual_merges = sum(1 for n, c in canonical.items() if n != c)
        logger.info(f"Merged {actual_merges:,} nodes into existing nodes")

        return SynthGraph(new_graph, new_positions)

    def _spatial_hash(self, position) -> Tuple[int, int]:
        """Mirrors ``GraphNode._spatial_hash``."""
        return (
            int(position[0] // self.grid_size),
            int(position[1] // self.grid_size),
        )
