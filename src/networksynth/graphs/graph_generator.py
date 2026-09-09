# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import random as rng
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from numpy import ndarray

from networksynth.configs import SynthParams
from networksynth.utils import build_graph, calculate_frame, progress, tagged

from ._graph_node import GraphNode
from .synth_graph import SynthGraph

logger = logging.getLogger(__name__)


FrontierDescriptor = namedtuple(
    "FrontierDescriptor",
    ["position", "degree", "base_angle", "clockwise", "parent_position"],
)


class GraphGenerator:

    def __init__(self, attributes_calculator, params: SynthParams):
        self._params = params
        GraphNode.initialize(attributes_calculator, self._params)

    def generate_network(
        self, frame_range: Optional[Tuple[int, int]] = None, regenerate_times: int = 100
    ):
        frame_range = frame_range or self._params.synthetic_frame_size

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
        snapshot_round_interval: int = 10,
        frame_range: Optional[Tuple[int, int]] = None,
        regenerate_times: int = 100,
    ):
        frame_range = frame_range or self._params.synthetic_frame_size

        for _ in range(regenerate_times):
            nodes, edges = self._bfs_network(
                frame_range,
                snapshot_callback=snapshot_callback,
                snapshot_round_interval=snapshot_round_interval,
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

    @staticmethod
    def _bfs_network(
        frame_range: Tuple[int, int],
        *,
        snapshot_callback=None,
        snapshot_round_interval: int = 0,
    ) -> Tuple[set, set]:
        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        scaled_frame_range = (round(frame_range[0] * 1.1), round(frame_range[1] * 1.1))
        frame = calculate_frame(
            center_position=root_node.position, frame_range=scaled_frame_range
        )

        from collections import deque

        take_snapshots = snapshot_callback is not None and snapshot_round_interval > 0
        snapshot_idx = 0

        if take_snapshots:
            snapshot_callback(
                [n.position for n in node_set], set(edge_set), frame, snapshot_idx
            )
            snapshot_idx += 1

        node_queue = deque([root_node])
        # A round is one frontier: as many pops as the queue held when it
        # began.  Snapshots are counted in rounds here as they are in Phase 2,
        # so an interval means the same thing in both modes.
        pops_left = len(node_queue)
        rounds = 0

        while node_queue:
            current_node = node_queue.popleft()
            pops_left -= 1
            if (
                _within_frame(current_node.position, frame)
                and current_node.generate_children()
            ):
                for child in current_node.children:
                    if child != current_node:
                        node_set.add(child)
                        edge_set.add((current_node.position, child.position))
                        node_queue.append(child)

            if pops_left > 0:
                continue

            rounds += 1
            pops_left = len(node_queue)
            if take_snapshots and rounds % snapshot_round_interval == 0:
                result = snapshot_callback(
                    [n.position for n in node_set], set(edge_set), frame, snapshot_idx
                )
                if result is False:
                    return node_set, edge_set
                snapshot_idx += 1

        if take_snapshots:
            snapshot_callback(
                [n.position for n in node_set], set(edge_set), frame, snapshot_idx
            )

        return node_set, edge_set

    @staticmethod
    def _bfs_network_with_frontier(
        frame_range: Tuple[int, int],
    ) -> Tuple[set, set, List, List[Tuple[float, float]], list]:
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
        expected_nodes: int = 0,
    ) -> SynthGraph:
        GraphNode.reset()

        take_snapshots = snapshot_callback is not None and snapshot_round_interval != 0

        def _fire_snapshot(idx):
            positions = [
                n.position for cell in GraphNode.node_grid.values() for n in cell
            ]
            edges = list({e for cell in GraphNode.edge_grid.values() for e in cell})
            snapshot_callback(positions, edges, global_frame, idx)

        frontier_positions: set = set()
        for tile in tile_data_list:
            for desc in tile["frontier"]:
                frontier_positions.add(desc.position)

        position_to_node: Dict[Tuple[float, float], GraphNode] = {}
        total_edges = 0

        all_edges: set = set()
        for tile in tile_data_list:
            for pos in tile["positions"]:
                if pos not in position_to_node and pos not in frontier_positions:
                    node = GraphNode.create_interior_node(pos)
                    position_to_node[pos] = node
            for edge in tile["edges"]:
                GraphNode.register_edge(edge)
                all_edges.add(edge)
                total_edges += 1

        logger.info(
            f"Phase 2: assembled {len(position_to_node):,} interior nodes, "
            f"{total_edges:,} edges in edge_grid",
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
            virtual_round = 0
            pops_this_round = 0
            round_size = len(frontier)

            while frontier:
                if _interrupted:
                    logger.info(
                        f"Phase 2 interrupted at round {virtual_round}.",
                        extra=tagged("PHASE2"),
                    )
                    break

                # A uniformly random pop, not the next in line: round-based BFS
                # lays a concentric shell per round and leaves a visible ripple
                # around every seed. Rounds survive as a pop budget only.
                last = len(frontier) - 1
                idx = rng.randint(0, last)
                if idx != last:
                    frontier[idx], frontier[last] = frontier[last], frontier[idx]
                node = frontier.pop()
                pops_this_round += 1

                if (
                    _within_frame(node.position, global_frame)
                    and node.generate_children()
                ):
                    for child in node.children:
                        if child != node and id(child) not in seen_frontier_ids:
                            seen_frontier_ids.add(id(child))
                            frontier.append(child)

                if pops_this_round >= round_size:
                    if not frontier:
                        logger.info(
                            f"Phase 2 round {virtual_round}: converged. "
                            f"{GraphNode.counts()}",
                            extra=tagged("PHASE2"),
                        )
                        break

                    if (virtual_round + 1) % 10 == 0 or virtual_round == 0:
                        logger.info(
                            f"Phase 2 round {virtual_round}: "
                            f"frontier={len(frontier):,}, {GraphNode.counts()}",
                            extra=tagged(
                                "PHASE2",
                                **progress(GraphNode.nodes_created(), expected_nodes),
                            ),
                        )

                    if take_snapshots and (virtual_round + 1) >= next_snapshot_round:
                        _fire_snapshot(snapshot_idx)
                        snapshot_idx += 1
                        if snapshot_log:
                            next_snapshot_round = max(
                                next_snapshot_round + 1,
                                int(next_snapshot_round * log_multiplier),
                            )
                        else:
                            next_snapshot_round += snapshot_round_interval

                    virtual_round += 1
                    if virtual_round >= max_rounds:
                        break
                    pops_this_round = 0
                    round_size = len(frontier)
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

        logger.info(
            f"Phase 2 complete: {len(all_nodes):,} nodes, "
            f"{len(all_edges):,} edges, {GraphNode.counts()}",
            extra=tagged("PHASE2"),
        )

        graph = SynthGraph.from_graph_nodes(all_nodes, all_edges)
        filtered = _filter_to_frame(graph, global_frame)
        return filtered


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
    positions = graph.positions()
    nodes_to_keep = set()
    for node in graph.nodes():
        pos = positions[node]
        if _within_frame(pos, frame):
            nodes_to_keep.add(node)
    return graph.subgraph(nodes_to_keep)
