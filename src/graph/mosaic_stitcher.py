# src/graph/mosaic_stitcher.py
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
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from utils import largest_connected_component

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
        # path compression
        while self.parent.get(x, x) != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # union by rank
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
        """
        Parameters
        ----------
        merge_threshold : float
            Maximum Euclidean distance between two nodes (from *different*
            tiles) for them to be merged.  Should equal
            ``avg_length * CLOSED_NODES_FACTOR`` to mirror the GraphNode
            merging behaviour used during generation.
        """
        self.merge_threshold = merge_threshold
        # spatial-hash cell size — same as GraphNode._grid_size
        self.grid_size = merge_threshold

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def stitch(self, tile_graphs: Dict[Tuple[int, int], nx.Graph]) -> nx.Graph:
        """
        Stitch all tile graphs into one large graph.

        Parameters
        ----------
        tile_graphs : dict[(row, col), nx.Graph]
            Each graph must already have its node positions in **global**
            coordinates (i.e. offset by the tile's grid position).

        Returns
        -------
        nx.Graph
            The stitched network (largest connected component, re-indexed).
        """
        logger.info(f"Stitching {len(tile_graphs)} tiles …")

        # 1. Merge all tiles into a single nx.Graph with unique node ids
        combined, node_tile_map = self._combine_tiles(tile_graphs)
        logger.info(
            f"Combined graph: {combined.number_of_nodes():,} nodes, "
            f"{combined.number_of_edges():,} edges"
        )

        # 2. Merge close cross-tile nodes
        merged = self._merge_close_nodes(combined, node_tile_map)
        logger.info(
            f"After merge:    {merged.number_of_nodes():,} nodes, "
            f"{merged.number_of_edges():,} edges"
        )

        # 3. Keep largest connected component
        result = largest_connected_component(merged, reindex=True)
        logger.info(
            f"Final network:  {result.number_of_nodes():,} nodes, "
            f"{result.number_of_edges():,} edges"
        )
        return result

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _combine_tiles(
        tile_graphs: Dict[Tuple[int, int], nx.Graph],
    ) -> Tuple[nx.Graph, Dict[int, Tuple[int, int]]]:
        """
        Merge every tile graph into *one* graph, giving each node a
        globally unique integer id.

        Returns
        -------
        combined : nx.Graph
        node_tile_map : dict[int, (row, col)]
            Maps each global node id to the tile it originated from.
        """
        combined = nx.Graph()
        node_tile_map: Dict[int, Tuple[int, int]] = {}
        counter = 0

        for (row, col), tile_graph in tile_graphs.items():
            positions = nx.get_node_attributes(tile_graph, "pos")
            old_to_new: Dict[int, int] = {}

            for old_id in tile_graph.nodes():
                new_id = counter
                combined.add_node(new_id, pos=positions[old_id])
                node_tile_map[new_id] = (row, col)
                old_to_new[old_id] = new_id
                counter += 1

            for u, v in tile_graph.edges():
                combined.add_edge(old_to_new[u], old_to_new[v])

        return combined, node_tile_map

    def _merge_close_nodes(
        self,
        graph: nx.Graph,
        node_tile_map: Dict[int, Tuple[int, int]],
    ) -> nx.Graph:
        """
        Find cross-tile node pairs closer than ``merge_threshold`` and
        merge them with a Union-Find structure.

        Algorithm
        ---------
        1. Build a spatial hash of all node positions (cell size =
           ``merge_threshold``, mirroring ``GraphNode._grid_size``).
        2. For every node, scan the 3×3 neighbourhood of cells
           (mirroring ``GraphNode._find_close_node``).
        3. Collect all cross-tile pairs within threshold, sort by
           distance (closest-first), and union them.
        4. Contract the graph: redirect every edge to its canonical
           representative, drop self-loops, and keep one copy of each
           multi-edge.
        """
        positions = nx.get_node_attributes(graph, "pos")

        # --- spatial hash ---
        spatial_hash: Dict[Tuple[int, int], list] = defaultdict(list)
        for node, pos in positions.items():
            key = self._spatial_hash(pos)
            spatial_hash[key].append(node)

        # --- discover merge candidates ---
        merge_pairs = []
        seen: set = set()

        for key, cell_nodes in spatial_hash.items():
            # 3×3 neighbourhood (same as GraphNode._find_close_node)
            neighbor_keys = [
                (key[0] + dx, key[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)
            ]
            neighbor_nodes = []
            for nk in neighbor_keys:
                neighbor_nodes.extend(spatial_hash.get(nk, []))

            for node_a in cell_nodes:
                tile_a = node_tile_map[node_a]
                pos_a = np.array(positions[node_a])

                for node_b in neighbor_nodes:
                    if node_a >= node_b:
                        continue  # avoid duplicates & self
                    if node_tile_map[node_b] == tile_a:
                        continue  # same tile → skip

                    pair_key = (node_a, node_b)
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    pos_b = np.array(positions[node_b])
                    dist = float(np.linalg.norm(pos_a - pos_b))
                    if dist < self.merge_threshold:
                        merge_pairs.append((dist, node_a, node_b))

        logger.info(f"Found {len(merge_pairs):,} cross-tile node pairs to merge")

        # sort closest-first so Union-Find representatives stay local
        merge_pairs.sort()

        # --- union-find ---
        uf = _UnionFind()
        for _, a, b in merge_pairs:
            uf.union(a, b)

        # --- contract graph ---
        canonical = {node: uf.find(node) for node in graph.nodes()}

        merged = nx.Graph()
        for node in graph.nodes():
            rep = canonical[node]
            if rep not in merged:
                merged.add_node(rep, pos=positions[rep])

        for u, v in graph.edges():
            cu, cv = canonical[u], canonical[v]
            if cu != cv:
                merged.add_edge(cu, cv)

        actual_merges = sum(1 for n, c in canonical.items() if n != c)
        logger.info(f"Merged {actual_merges:,} nodes into existing nodes")

        return merged

    def _spatial_hash(self, position) -> Tuple[int, int]:
        """Mirrors ``GraphNode._spatial_hash``."""
        return (
            int(position[0] // self.grid_size),
            int(position[1] // self.grid_size),
        )
