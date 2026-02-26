# src/graph/graph_generator.py

import logging
import random as rng
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from numpy import ndarray

from config import BaseConfig
from utils import build_graph, calculate_frame

from ._graph_node import GraphNode
from .synth_graph import SynthGraph

logger = logging.getLogger(__name__)

FrontierDescriptor = namedtuple(
    "FrontierDescriptor",
    ["position", "degree", "base_angle", "clockwise", "parent_position"],
)


class GraphGenerator:

    def __init__(self, attributes_calculator):
        GraphNode.initialize(attributes_calculator)

    def generate_network(
        self, frame_range: Optional[Tuple[int, int]] = None, regenerate_times: int = 100
    ):
        frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE

        for _ in range(regenerate_times):
            nodes, edges = self._bfs_network(frame_range)
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception(
                "Failed to generate a original_network within the specified attempts."
            )

        _synthetic_network = build_graph(nodes, edges, arg_type="graph_node")
        frame = calculate_frame(graph=_synthetic_network, frame_range=frame_range)
        synthetic_network = _filter_graph(_synthetic_network, frame)
        return synthetic_network

    def generate_scaled_network(
        self,
        scale_rows: int,
        scale_cols: int,
        max_rounds: int = 500,
        root_spacing_factor: float = 0.7,
    ) -> SynthGraph:
        """Generate a large network via synchronized multi-root BFS.

        Places ``scale_rows × scale_cols`` root nodes on a grid and grows
        them all simultaneously, one generation at a time, with random
        iteration order per round.  Components merge naturally through the
        existing close-node logic.

        ``root_spacing_factor`` controls root distance relative to
        ``SYNTHETIC_FRAME_SIZE``.  Values < 1.0 make components overlap
        sooner, promoting cross-root merges.
        """
        frame_w, frame_h = BaseConfig.SYNTHETIC_FRAME_SIZE
        spacing_w = frame_w * root_spacing_factor
        spacing_h = frame_h * root_spacing_factor

        root_positions: List[Tuple[float, float]] = []
        for r in range(scale_rows):
            for c in range(scale_cols):
                x = c * spacing_w + spacing_w / 2.0
                y = r * spacing_h + spacing_h / 2.0
                root_positions.append((x, y))

        total_w = scale_cols * spacing_w
        total_h = scale_rows * spacing_h
        margin_x = spacing_w * 0.5
        margin_y = spacing_h * 0.5
        global_frame = (
            (-margin_x, total_w + margin_x),
            (-margin_y, total_h + margin_y),
        )

        logger.info(
            f"Scaled generation: {scale_rows}×{scale_cols} roots, "
            f"spacing=({spacing_w:.0f}, {spacing_h:.0f}) "
            f"(factor={root_spacing_factor}), "
            f"global_frame={global_frame}"
        )

        nodes, edges = self._multi_root_bfs(root_positions, global_frame, max_rounds)

        if not nodes or len(nodes) < 100:
            raise RuntimeError(
                f"Scaled generation produced only {len(nodes) if nodes else 0} nodes"
            )

        synthetic_network = SynthGraph.from_graph_nodes(nodes, edges)
        filtered = _filter_to_frame(synthetic_network, global_frame)
        return filtered

    @staticmethod
    def _bfs_network(frame_range: Optional[Tuple[int, int]] = None) -> Tuple[set, set]:
        frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE

        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        scaled_frame_range = (round(frame_range[0] * 1.1), round(frame_range[1] * 1.1))
        frame = calculate_frame(
            center_position=root_node.position, frame_range=scaled_frame_range
        )

        from collections import deque

        node_queue = deque([root_node])
        while node_queue:
            current_node = node_queue.popleft()
            if not _within_frame(current_node.position, frame):
                continue
            if current_node.generate_children():
                for child in current_node.children:
                    if child != current_node:
                        node_set.add(child)
                        edge_set.add((current_node.position, child.position))
                        node_queue.append(child)
        return node_set, edge_set

    @staticmethod
    def _bfs_network_with_frontier(
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[set, set, List, List[Tuple[float, float]], list]:
        """BFS that also captures frontier descriptors for Phase-2 continuation.

        Returns
        -------
        inner_nodes : set[GraphNode]
            Nodes inside the generation frame (for quality checking).
        inner_edges : set
            Edges whose *both* endpoints are inside the generation frame.
        frontier_descriptors : list[FrontierDescriptor]
            Serializable descriptors for nodes just outside the frame that
            can be continued in Phase 2.
        all_positions : list[tuple[float, float]]
            Every node's position (inner + margin + outside).
        all_edges : list[tuple]
            Every edge tuple (inner + boundary).
        """
        frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE

        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        scaled_frame_range = (
            round(frame_range[0] * 1.1),
            round(frame_range[1] * 1.1),
        )
        frame = calculate_frame(
            center_position=root_node.position, frame_range=scaled_frame_range
        )

        from collections import deque

        node_queue = deque([root_node])
        while node_queue:
            current_node = node_queue.popleft()
            if not _within_frame(current_node.position, frame):
                continue
            if current_node.generate_children():
                for child in current_node.children:
                    if child != current_node:
                        node_set.add(child)
                        edge_set.add((current_node.position, child.position))
                        node_queue.append(child)

        inner_nodes: set = set()
        frontier_descriptors: list = []
        for node in node_set:
            if _within_frame(node.position, frame):
                inner_nodes.add(node)
            elif node.degree > 1 and len(node.children) <= 1:
                frontier_descriptors.append(
                    FrontierDescriptor(
                        position=node.position,
                        degree=node.degree,
                        base_angle=node.base_angle,
                        clockwise=node.clockwise,
                        parent_position=(node.parent.position if node.parent else None),
                    )
                )

        inner_edges = {
            e
            for e in edge_set
            if _within_frame(e[0], frame) and _within_frame(e[1], frame)
        }
        all_positions = [node.position for node in node_set]
        all_edge_tuples = list(edge_set)

        return (
            inner_nodes,
            inner_edges,
            frontier_descriptors,
            all_positions,
            all_edge_tuples,
        )

    @staticmethod
    def assemble_and_continue(
        tile_data_list: List[Dict],
        global_frame: Tuple[Tuple[float, float], Tuple[float, float]],
        max_rounds: int,
    ) -> SynthGraph:
        """Phase 2: populate grids from tiles, then BFS from frontier nodes.

        Frozen tile nodes are kept OUT of ``node_grid`` so that frontier
        branches cannot wastefully merge back into their own tile's
        interior.  Tile edges ARE added to ``edge_grid``, which is
        sufficient to prevent new growth from crossing existing tiles.

        Only frontier nodes (and newly grown children) enter ``node_grid``,
        so the close-node merge logic only fires between actively growing
        branches — exactly the cross-tile connections we want.

        Each element of *tile_data_list* must have:
          - ``positions``: list of (x, y) in **global** coordinates
          - ``edges``:     list of ((x1,y1),(x2,y2)) in global coordinates
          - ``frontier``:  list of :class:`FrontierDescriptor` in global coords
        """
        GraphNode.reset()
        GraphNode._merge_priority = True

        frontier_positions: set = set()
        for tile in tile_data_list:
            for desc in tile["frontier"]:
                frontier_positions.add(desc.position)

        position_to_node: Dict[Tuple[float, float], GraphNode] = {}
        total_edges = 0

        for tile in tile_data_list:
            for pos in tile["positions"]:
                if pos not in position_to_node and pos not in frontier_positions:
                    frozen = GraphNode.create_frozen(pos, register_in_grid=False)
                    position_to_node[pos] = frozen
            for edge in tile["edges"]:
                GraphNode.add_frozen_edge(edge)
                total_edges += 1

        logger.info(
            f"Phase 2: assembled {len(position_to_node):,} frozen nodes "
            f"(not in node_grid), {total_edges:,} edges in edge_grid"
        )

        all_nodes: set = set(position_to_node.values())
        all_edges: set = set()
        for tile in tile_data_list:
            for edge in tile["edges"]:
                all_edges.add(edge)
        seen_ids: set = {id(n) for n in all_nodes}

        frontier: list = []
        for tile in tile_data_list:
            for desc in tile["frontier"]:
                parent = position_to_node.get(desc.parent_position)
                if parent is None:
                    parent = GraphNode.create_frozen(
                        desc.parent_position, register_in_grid=False
                    )
                    position_to_node[desc.parent_position] = parent
                    all_nodes.add(parent)
                    seen_ids.add(id(parent))

                fnode = GraphNode.create_frontier_node(
                    desc.position,
                    desc.degree,
                    desc.base_angle,
                    desc.clockwise,
                    parent,
                )
                position_to_node[desc.position] = fnode
                frontier.append(fnode)
                all_nodes.add(fnode)
                seen_ids.add(id(fnode))

        logger.info(f"Phase 2: {len(frontier):,} frontier nodes ready for expansion")

        for round_num in range(max_rounds):
            rng.shuffle(frontier)
            next_frontier: list = []

            for node in frontier:
                if not _within_frame(node.position, global_frame):
                    continue
                if node.generate_children():
                    for child in node.children:
                        all_nodes.add(child)
                        all_edges.add((node.position, child.position))
                        if id(child) not in seen_ids:
                            seen_ids.add(id(child))
                            next_frontier.append(child)

            if not next_frontier:
                logger.info(
                    f"Phase 2 round {round_num}: converged. "
                    f"Total: {len(all_nodes):,} nodes, {len(all_edges):,} edges"
                )
                break

            if (round_num + 1) % 10 == 0 or round_num == 0:
                logger.info(
                    f"Phase 2 round {round_num}: frontier={len(next_frontier):,}, "
                    f"nodes={len(all_nodes):,}, edges={len(all_edges):,}, "
                    f"merged={GraphNode._merged_edge:,}, "
                    f"aborted={GraphNode._aborted_edge:,}"
                )

            frontier = next_frontier

        GraphNode._merge_priority = False
        logger.info(
            f"Phase 2 complete: {len(all_nodes):,} nodes, "
            f"{len(all_edges):,} edges, "
            f"merged={GraphNode._merged_edge:,}, "
            f"aborted={GraphNode._aborted_edge:,}"
        )

        graph = SynthGraph.from_graph_nodes(all_nodes, all_edges)
        filtered = _filter_to_frame(graph, global_frame)
        return filtered

    @staticmethod
    def _multi_root_bfs(
        root_positions: List[Tuple[float, float]],
        global_frame: Tuple[Tuple[float, float], Tuple[float, float]],
        max_rounds: int,
    ) -> Tuple[set, set]:
        """Synchronized multi-root BFS.

        All roots share a single ``node_grid`` / ``edge_grid`` so the
        standard close-node and edge-intersection checks work across
        roots automatically.  Each round shuffles the frontier randomly
        before expanding, ensuring no systematic bias.
        """
        GraphNode.reset()
        GraphNode._merge_priority = True

        all_nodes: set = set()
        all_edges: set = set()
        seen_ids: set = set()

        # --- Phase 1: create every root (each also spawns its first child) ---
        frontier: list = []
        n_roots = len(root_positions)
        for i, pos in enumerate(root_positions):
            root = GraphNode(pos)
            all_nodes.add(root)
            seen_ids.add(id(root))
            frontier.append(root)

            for child in root.children:
                all_nodes.add(child)
                all_edges.add((root.position, child.position))
                if id(child) not in seen_ids:
                    seen_ids.add(id(child))
                    frontier.append(child)

            if (i + 1) % max(1, n_roots // 10) == 0:
                logger.info(f"Root init: {i + 1:,}/{n_roots:,}")

        logger.info(
            f"Initialized {n_roots:,} roots → "
            f"{len(all_nodes):,} nodes, {len(all_edges):,} edges, "
            f"frontier={len(frontier):,}"
        )

        # --- Phase 2: round-by-round expansion ---
        for round_num in range(max_rounds):
            rng.shuffle(frontier)
            next_frontier: list = []

            for node in frontier:
                if not _within_frame(node.position, global_frame):
                    continue
                if node.generate_children():
                    for child in node.children:
                        all_nodes.add(child)
                        all_edges.add((node.position, child.position))
                        if id(child) not in seen_ids:
                            seen_ids.add(id(child))
                            next_frontier.append(child)

            if not next_frontier:
                logger.info(
                    f"Round {round_num}: converged (no new nodes). "
                    f"Total: {len(all_nodes):,} nodes, {len(all_edges):,} edges"
                )
                break

            if (round_num + 1) % 10 == 0 or round_num == 0:
                logger.info(
                    f"Round {round_num}: frontier={len(next_frontier):,}, "
                    f"nodes={len(all_nodes):,}, edges={len(all_edges):,}, "
                    f"merged={GraphNode._merged_edge:,}, "
                    f"aborted={GraphNode._aborted_edge:,}"
                )

            frontier = next_frontier

        GraphNode._merge_priority = False
        logger.info(
            f"Multi-root BFS complete: {len(all_nodes):,} nodes, "
            f"{len(all_edges):,} edges, "
            f"merged={GraphNode._merged_edge:,}, "
            f"aborted={GraphNode._aborted_edge:,}"
        )
        return all_nodes, all_edges


def _within_frame(
    position: Union[list, tuple, ndarray], frame: Union[list, tuple, ndarray]
) -> bool:
    x, y = position
    return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]


def _filter_graph(graph: SynthGraph, frame: Union[list, tuple]) -> SynthGraph:
    positions = graph.positions()
    nodes_to_keep = set()
    for node in graph.nodes():
        pos = positions[node]
        if _within_frame(pos, frame):
            nodes_to_keep.add(node)
    filtered = graph.subgraph(nodes_to_keep)
    return filtered.largest_connected_component()


def _filter_to_frame(graph: SynthGraph, frame: Union[list, tuple]) -> SynthGraph:
    """Keep all nodes within the frame (no LCC filtering).

    Used by the scaling mode where multiple root components may not
    fully merge but should all be retained.
    """
    positions = graph.positions()
    nodes_to_keep = set()
    for node in graph.nodes():
        pos = positions[node]
        if _within_frame(pos, frame):
            nodes_to_keep.add(node)
    return graph.subgraph(nodes_to_keep)
