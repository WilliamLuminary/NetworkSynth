# src/graphs/graph_generator.py

import logging
import random as rng
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from numpy import ndarray

from configs import BaseConfig
from configs.base_config import tagged
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

    def generate_network_with_snapshots(
        self,
        snapshot_callback,
        snapshot_interval: int = 50,
        frame_range: Optional[Tuple[int, int]] = None,
        regenerate_times: int = 100,
    ):
        """Generate a network while taking periodic BFS snapshots.

        Parameters
        ----------
        snapshot_callback : callable
            ``callback(positions, edges, frame, step_index)`` where
            *positions* is a list of ``(x, y)`` tuples and *edges* is a
            set of ``((x1, y1), (x2, y2))`` tuples.
        snapshot_interval : int
            Take a snapshot every time the network grows by this many
            nodes.
        """
        frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE

        for _ in range(regenerate_times):
            nodes, edges = self._bfs_network(
                frame_range,
                snapshot_callback=snapshot_callback,
                snapshot_interval=snapshot_interval,
            )
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception(
                "Failed to generate a network within the specified attempts."
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
            f"global_frame={global_frame}",
            extra=tagged("PHASE1"),
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
    def _bfs_network(
        frame_range: Optional[Tuple[int, int]] = None,
        *,
        snapshot_callback=None,
        snapshot_interval: int = 0,
    ) -> Tuple[set, set]:
        frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE

        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        scaled_frame_range = (round(frame_range[0] * 1.1), round(frame_range[1] * 1.1))
        frame = calculate_frame(
            center_position=root_node.position, frame_range=scaled_frame_range
        )

        from collections import deque

        take_snapshots = snapshot_callback is not None and snapshot_interval > 0
        snapshot_idx = 0

        if take_snapshots:
            snapshot_callback(
                [n.position for n in node_set], set(edge_set), frame, snapshot_idx
            )
            snapshot_idx += 1

        next_snapshot_at = snapshot_interval if take_snapshots else float("inf")

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

            if take_snapshots and len(node_set) >= next_snapshot_at:
                result = snapshot_callback(
                    [n.position for n in node_set], set(edge_set), frame, snapshot_idx
                )
                if result is False:
                    return node_set, edge_set
                snapshot_idx += 1
                next_snapshot_at += snapshot_interval

        if take_snapshots:
            snapshot_callback(
                [n.position for n in node_set], set(edge_set), frame, snapshot_idx
            )

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
        *,
        snapshot_callback=None,
        snapshot_round_interval: int = 0,
    ) -> SynthGraph:
        """Phase 2: populate grids from tiles, then BFS from frontier nodes.

        Tile edges are added to ``edge_grid`` and interior nodes to
        ``node_grid`` so the same close-node merge and close-edge
        avoidance logic used in Phase 1 applies identically here.

        Uses the same ``_place_child`` logic as Phase 1 to ensure
        gap-fill regions have the same distributional properties as
        tile interiors.

        Each element of *tile_data_list* must have:
          - ``positions``: list of (x, y) in **global** coordinates
          - ``edges``:     list of ((x1,y1),(x2,y2)) in global coordinates
          - ``frontier``:  list of :class:`FrontierDescriptor` in global coords

        Parameters
        ----------
        snapshot_callback : callable, optional
            ``callback(positions, edges, global_frame, index)`` called
            after assembly and every *snapshot_round_interval* rounds.
        snapshot_round_interval : int
            >0: fire *snapshot_callback* every N rounds (linear).
            <0: fire ~|N| log-spaced snapshots across all rounds.
            0: disable snapshots.
        """
        GraphNode.reset()

        take_snapshots = snapshot_callback is not None and snapshot_round_interval != 0

        # --- Edge-grid policy ---------------------------------------- #
        # All tile NODES go into node_grid so that close-node merging
        # works correctly (prevents overlap when frontiers reach
        # neighbouring tile interiors).
        #
        # Pre-existing tile EDGES are kept OUT of edge_grid entirely.
        # Phase 2 frontier expansion builds its own edge environment
        # incrementally — the same way Phase 1 does during BFS.  This
        # ensures _any_close_edge and _check_intersection see a similar
        # density to Phase 1, preventing the "tight / even-fill" artefact
        # that occurs when the grid is pre-loaded with fully-developed
        # tile boundary structure.
        tile_edges: set = set()

        def _fire_snapshot(idx):
            positions = [
                n.position for cell in GraphNode.node_grid.values() for n in cell
            ]
            edges = list({e for cell in GraphNode.edge_grid.values() for e in cell})
            edges.extend(tile_edges)
            snapshot_callback(positions, edges, global_frame, idx)

        frontier_positions: set = set()
        for tile in tile_data_list:
            for desc in tile["frontier"]:
                frontier_positions.add(desc.position)

        position_to_node: Dict[Tuple[float, float], GraphNode] = {}

        for tile in tile_data_list:
            for pos in tile["positions"]:
                if pos not in position_to_node and pos not in frontier_positions:
                    node = GraphNode.create_interior_node(pos)
                    position_to_node[pos] = node
            for edge in tile["edges"]:
                tile_edges.add(edge)

        logger.info(
            f"Phase 2: assembled {len(position_to_node):,} interior nodes, "
            f"{len(tile_edges):,} tile edges (output-only, not in edge_grid)",
            extra=tagged("PHASE2"),
        )

        frontier: list = []
        seen_frontier_ids: set = set()
        for tile in tile_data_list:
            for desc in tile["frontier"]:
                parent = position_to_node.get(desc.parent_position)
                if parent is None:
                    parent = GraphNode.create_interior_node(desc.parent_position)
                    position_to_node[desc.parent_position] = parent

                fnode = GraphNode.create_frontier_node(
                    desc.position,
                    desc.degree,
                    desc.base_angle,
                    desc.clockwise,
                    parent,
                )
                position_to_node[desc.position] = fnode
                frontier.append(fnode)
                seen_frontier_ids.add(id(fnode))

        logger.info(
            f"Phase 2: {len(frontier):,} frontier nodes ready for expansion",
            extra=tagged("PHASE2"),
        )

        del position_to_node, frontier_positions

        snapshot_idx = 0
        if take_snapshots:
            _fire_snapshot(snapshot_idx)
            snapshot_idx += 1

        snapshot_log = snapshot_round_interval < 0
        if snapshot_log:
            desired_count = abs(snapshot_round_interval)
            log_multiplier = (
                max_rounds ** (1.0 / desired_count) if desired_count > 1 else 2.0
            )
            next_snapshot_round = 1
            logger.info(
                f"Log snapshot schedule: ~{desired_count} snapshots, "
                f"multiplier={log_multiplier:.3f}",
                extra=tagged("SNAPSHOT"),
            )
        else:
            log_multiplier = 1.0
            next_snapshot_round = snapshot_round_interval

        import signal

        _interrupted = False
        _prev_handler = signal.getsignal(signal.SIGINT)

        def _on_sigint(sig, frame):
            nonlocal _interrupted
            _interrupted = True

        signal.signal(signal.SIGINT, _on_sigint)

        try:
            for round_num in range(max_rounds):
                if _interrupted:
                    logger.info(
                        f"Phase 2 interrupted at round {round_num}.",
                        extra=tagged("PHASE2"),
                    )
                    break

                rng.shuffle(frontier)
                next_frontier: list = []

                for node in frontier:
                    if not _within_frame(node.position, global_frame):
                        continue
                    if node.generate_children():
                        for child in node.children:
                            if child != node and id(child) not in seen_frontier_ids:
                                seen_frontier_ids.add(id(child))
                                next_frontier.append(child)

                if not next_frontier:
                    logger.info(
                        f"Phase 2 round {round_num}: converged. "
                        f"merged={GraphNode._merged_edge:,}, "
                        f"aborted={GraphNode._aborted_edge:,}",
                        extra=tagged("PHASE2"),
                    )
                    break

                if (round_num + 1) % 10 == 0 or round_num == 0:
                    logger.info(
                        f"Phase 2 round {round_num}: frontier={len(next_frontier):,}, "
                        f"merged={GraphNode._merged_edge:,}, "
                        f"aborted={GraphNode._aborted_edge:,}",
                        extra=tagged("PHASE2"),
                    )

                frontier = next_frontier

                if take_snapshots and (round_num + 1) >= next_snapshot_round:
                    _fire_snapshot(snapshot_idx)
                    snapshot_idx += 1
                    if snapshot_log:
                        next_snapshot_round = max(
                            next_snapshot_round + 1,
                            int(next_snapshot_round * log_multiplier),
                        )
                    else:
                        next_snapshot_round += snapshot_round_interval
        finally:
            signal.signal(signal.SIGINT, _prev_handler)

        if _interrupted:
            raise KeyboardInterrupt

        del seen_frontier_ids

        if take_snapshots:
            _fire_snapshot(snapshot_idx)

        all_nodes: set = set()
        for cell_nodes in GraphNode.node_grid.values():
            all_nodes.update(cell_nodes)

        all_edges: set = set()
        for cell_edges in GraphNode.edge_grid.values():
            all_edges.update(cell_edges)
        all_edges.update(tile_edges)

        logger.info(
            f"Phase 2 complete: {len(all_nodes):,} nodes, "
            f"{len(all_edges):,} edges, "
            f"merged={GraphNode._merged_edge:,}, "
            f"aborted={GraphNode._aborted_edge:,}",
            extra=tagged("PHASE2"),
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
                logger.info(f"Root init: {i + 1:,}/{n_roots:,}", extra=tagged("BFS"))

        logger.info(
            f"Initialized {n_roots:,} roots → "
            f"{len(all_nodes):,} nodes, {len(all_edges):,} edges, "
            f"frontier={len(frontier):,}",
            extra=tagged("BFS"),
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
                    f"Total: {len(all_nodes):,} nodes, {len(all_edges):,} edges",
                    extra=tagged("BFS"),
                )
                break

            if (round_num + 1) % 10 == 0 or round_num == 0:
                logger.info(
                    f"Round {round_num}: frontier={len(next_frontier):,}, "
                    f"nodes={len(all_nodes):,}, edges={len(all_edges):,}, "
                    f"merged={GraphNode._merged_edge:,}, "
                    f"aborted={GraphNode._aborted_edge:,}",
                    extra=tagged("BFS"),
                )

            frontier = next_frontier

        logger.info(
            f"Multi-root BFS complete: {len(all_nodes):,} nodes, "
            f"{len(all_edges):,} edges, "
            f"merged={GraphNode._merged_edge:,}, "
            f"aborted={GraphNode._aborted_edge:,}",
            extra=tagged("BFS"),
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
