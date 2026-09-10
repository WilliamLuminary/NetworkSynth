# SPDX-License-Identifier: GPL-3.0-or-later
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from networksynth.configs import SynthParams

Position = Tuple[float, float]
Edge = Tuple[Position, Position]


@dataclass(frozen=True)
class TraversalRules:

    degree_dist: Dict[int, float]
    degree_trans_probs: dict
    degree_angles: dict
    degree_lengths: dict

    closed_nodes_thr_sq: float
    closed_edges_thr_sq: float
    grid_size: float
    node_search_radius: int
    edge_search_radius: int

    @classmethod
    def build(cls, attrs, params: SynthParams) -> "TraversalRules":
        avg_length = attrs.average_length
        closed_nodes_thr = avg_length * params.closed_nodes_factor
        closed_edges_thr = avg_length * params.closed_edges_factor
        return cls(
            degree_dist=attrs.degree_distribution,
            degree_trans_probs=attrs.degree_transition_probs,
            degree_angles=attrs.degree_angles,
            degree_lengths=attrs.degree_lengths,
            closed_nodes_thr_sq=closed_nodes_thr**2,
            closed_edges_thr_sq=closed_edges_thr**2,
            grid_size=avg_length,
            node_search_radius=math.ceil(params.closed_nodes_factor),
            edge_search_radius=math.ceil(params.closed_edges_factor),
        )


@dataclass(frozen=True)
class PlacementCounts:

    merged: int
    aborted: int

    def __str__(self) -> str:
        return f"merged={self.merged:,}, aborted={self.aborted:,}"


@dataclass
class Traversal:
    """Everything one growth pass records: its rules, the spatial grids that
    answer "what is near here", and the ids and counters it hands out. It is
    built for one pass and dropped with it, so nothing survives into the next."""

    rules: TraversalRules
    rng: np.random.Generator
    node_grid: Dict[Tuple[int, int], Set["GraphNode"]] = field(
        default_factory=lambda: defaultdict(set)
    )
    edge_grid: Dict[Tuple[int, int], Set[Edge]] = field(
        default_factory=lambda: defaultdict(set)
    )
    created: int = 0
    merged: int = 0
    aborted: int = 0

    @classmethod
    def build(cls, attrs, params: SynthParams, rng: np.random.Generator) -> "Traversal":
        return cls(TraversalRules.build(attrs, params), rng)

    def next_id(self) -> int:
        self.created += 1
        return self.created - 1

    def nodes_created(self) -> int:
        """Nodes made so far. The grid cannot say: a node is filed under its
        children's cells too, so cell sizes over-count."""
        return self.created

    def counts(self) -> PlacementCounts:
        return PlacementCounts(merged=self.merged, aborted=self.aborted)

    def cell(self, position: Position) -> Tuple[int, int]:
        grid_size = self.rules.grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    def cells_of(self, p1: Position, p2: Position) -> Set[Tuple[int, int]]:
        gs = self.rules.grid_size
        gx1 = int(p1[0] // gs)
        gx2 = int(p2[0] // gs)
        gy1 = int(p1[1] // gs)
        gy2 = int(p2[1] // gs)
        if gx1 > gx2:
            gx1, gx2 = gx2, gx1
        if gy1 > gy2:
            gy1, gy2 = gy2, gy1
        return {(x, y) for x in range(gx1, gx2 + 1) for y in range(gy1, gy2 + 1)}

    def file_node(self, node: "GraphNode", position: Position) -> None:
        self.node_grid[self.cell(position)].add(node)

    def register_edge(self, edge: Edge) -> None:
        for key in self.cells_of(*edge):
            self.edge_grid[key].add(edge)

    def all_nodes(self) -> Set["GraphNode"]:
        return {node for cell in self.node_grid.values() for node in cell}

    def all_edges(self) -> Set[Edge]:
        return {edge for cell in self.edge_grid.values() for edge in cell}

    def create_interior_node(self, position: Position) -> "GraphNode":
        node = object.__new__(GraphNode)
        node._t = self
        node.id = self.next_id()
        node.position = position
        node.degree = 2
        node.children = [None, None]
        node.parent = None
        node.clockwise = False
        node.base_angle = 0.0
        self.file_node(node, position)
        return node

    def create_frontier_node(
        self, position, degree, base_angle, clockwise, parent_node
    ) -> "GraphNode":
        node = object.__new__(GraphNode)
        node._t = self
        node.id = self.next_id()
        node.position = position
        node.degree = degree
        node.base_angle = base_angle
        node.clockwise = clockwise
        node.parent = parent_node
        node.children = [parent_node]
        self.file_node(node, position)
        return node


class GraphNode:

    def __init__(self, traversal: Traversal, position, parent=None, parent_angle=None):
        self._t = traversal
        self.id: int = traversal.next_id()

        self.position: Position = position
        self.clockwise: bool = bool(traversal.rng.integers(2))
        self.parent: GraphNode = parent
        self.children: List[GraphNode] = []
        if parent is not None and parent_angle is not None:
            self.degree = self._choose_degree_by_parent(parent.degree)
            self.base_angle = (parent_angle + 180) % 360
            self._add_child(self.parent)
        elif parent is None and parent_angle is None:
            self.degree = self._choose_degree_random()
            self.base_angle = float(traversal.rng.uniform(0, 360))
            self._initialize_root_node()
        else:
            raise ValueError(
                "Parent and parent_angle must be provided together or not at all."
            )

    def _choose_degree_by_parent(self, parent_degree) -> int:
        transitions = self._t.rules.degree_trans_probs[parent_degree]
        degrees = list(transitions.keys())
        probabilities = list(transitions.values())
        return int(self._t.rng.choice(degrees, p=probabilities))

    def _choose_degree_random(self) -> int:
        degrees = list(self._t.rules.degree_dist.keys())
        probabilities = list(self._t.rules.degree_dist.values())
        return int(self._t.rng.choice(degrees, p=probabilities))

    def _initialize_root_node(self) -> None:
        length = _sample(self._t.rng, self._t.rules.degree_lengths[self.degree], 1)[0]
        child_position = self._polar_to_cartesian([length], [self.base_angle])[0]
        child = GraphNode(
            self._t, child_position, parent=self, parent_angle=self.base_angle
        )
        self._add_child(child)
        self._t.file_node(self, self.position)
        self._t.file_node(self, child.position)
        self._t.register_edge((self.position, child_position))

    def _add_child(self, child) -> None:
        assert (
            len(self.children) < self.degree
        ), f"{self} cannot have more than {self.degree} children."
        self.children.append(child)
        self._t.file_node(self, child.position)

    def generate_children(self) -> bool:
        if len(self.children) > 1 or self.degree == 1:
            return False

        angles, lengths = self._generate_angles_and_lengths()
        children_positions = self._polar_to_cartesian(lengths, angles)
        assert len(children_positions) == self.degree - 1

        for child_position, angle in zip(children_positions, angles):
            self._place_child(child_position, angle)
        return True

    def _place_child(self, child_position, angle) -> None:
        close_node = self._get_closest_valid_node(child_position)
        if close_node is not None:
            new_edge = (self.position, close_node.position)
            if not self._check_intersection(new_edge):
                self._add_child(close_node)
                self._t.register_edge(new_edge)
                self._t.merged += 1
            else:
                self._t.aborted += 1
            return
        if self._any_close_edge(child_position):
            self._t.aborted += 1
            return
        new_edge = (self.position, child_position)
        if not self._check_intersection(new_edge):
            child_node = GraphNode(
                self._t, child_position, parent=self, parent_angle=angle
            )
            self._add_child(child_node)
            self._t.register_edge(new_edge)
            self._t.file_node(self, child_node.position)
        else:
            self._t.aborted += 1

    def _get_closest_valid_node(self, position):
        close_nodes_with_distances = self._find_close_node(position)
        if close_nodes_with_distances:
            return min(close_nodes_with_distances, key=lambda x: (x[1], x[0].id))[0]
        return None

    def _find_close_node(self, position):
        key = self._t.cell(position)
        px, py = position
        rules = self._t.rules
        thr_sq = rules.closed_nodes_thr_sq
        r = rules.node_search_radius
        node_grid = self._t.node_grid
        close_nodes_with_distances = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for node in node_grid.get((key[0] + dx, key[1] + dy), ()):
                    if node != self and node not in self.children:
                        ddx = node.position[0] - px
                        ddy = node.position[1] - py
                        dist_sq = ddx * ddx + ddy * ddy
                        if dist_sq < thr_sq:
                            close_nodes_with_distances.append((node, dist_sq))
        return close_nodes_with_distances

    def _any_close_edge(self, position) -> bool:
        key = self._t.cell(position)
        r = self._t.rules.edge_search_radius
        edge_grid = self._t.edge_grid
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for edge in edge_grid.get((key[0] + dx, key[1] + dy), ()):
                    if self._is_interfering_edge(edge, position):
                        return True
        return False

    def _is_interfering_edge(self, edge, position) -> bool:
        if self.parent and (self.parent.position in edge or self.position in edge):
            return False
        p1, p2 = edge
        thr_sq = self._t.rules.closed_edges_thr_sq
        dx1 = position[0] - p1[0]
        dy1 = position[1] - p1[1]
        if dx1 * dx1 + dy1 * dy1 < thr_sq:
            return True
        dx2 = position[0] - p2[0]
        dy2 = position[1] - p2[1]
        return dx2 * dx2 + dy2 * dy2 < thr_sq

    def _generate_angles_and_lengths(self) -> Tuple[list[float], ...]:
        if self.degree == 1:
            return [], []
        rules = self._t.rules
        raw = _sample(self._t.rng, rules.degree_angles[self.degree], self.degree - 1)
        sign = 1 if self.clockwise else -1
        base = self.base_angle
        acc = 0.0
        angles = []
        for a in raw:
            acc += sign * a
            angles.append(acc + base)
        lengths = _sample(
            self._t.rng, rules.degree_lengths[self.degree], self.degree - 1
        )
        return angles, lengths

    def _polar_to_cartesian(
        self, lengths: List[float], angles: List[float]
    ) -> List[Position]:
        x, y = self.position
        result = []
        for length, angle in zip(lengths, angles):
            angle_rad = math.radians(angle)
            result.append(
                (
                    x + length * math.cos(angle_rad),
                    y + length * math.sin(angle_rad),
                )
            )
        return result

    def _check_intersection(self, new_edge: Edge) -> bool:
        edge_grid = self._t.edge_grid
        for cell in self._t.cells_of(*new_edge):
            for edge in edge_grid.get(cell, ()):
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue
                if _do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    def __repr__(self):
        return f"GraphNode(id_counter={self.id})"

    def __str__(self):
        return f"Node No. {self.id}"


def _sample(rng: np.random.Generator, population: list, k: int) -> list:
    return [population[i] for i in rng.integers(len(population), size=k)]


def _do_intersect(p1: Position, q1: Position, p2: Position, q2: Position) -> bool:
    def _orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if val == 0 else 1 if val > 0 else 2

    def _on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
            1
        ] <= max(p[1], r[1])

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    return (
        (o1 != o2 and o3 != o4)
        or (o1 == 0 and _on_segment(p1, p2, q1))
        or (o2 == 0 and _on_segment(p1, q2, q1))
        or (o3 == 0 and _on_segment(p2, p1, q2))
        or (o4 == 0 and _on_segment(p2, q1, q2))
    )
