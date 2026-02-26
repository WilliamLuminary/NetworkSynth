# src/graph/_graph_node.py
import itertools
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

from config import BaseConfig


class GraphNode:
    id_counter = None
    node_grid: Dict[tuple[float, float], set]
    edge_grid: Dict[
        Tuple[int, int], Set[Tuple[Tuple[float, float], Tuple[float, float]]]
    ]
    _aborted_edge: int
    _merged_edge: int

    # Class-level properties
    _degree_dist = None
    _degree_trans_probs = None
    _degree_angles = None
    _degree_lengths = None
    _avg_length: float

    # Class-level Parameters
    _closed_nodes_thr: float
    _closed_edges_thr: float
    _grid_size: float

    _closed_nodes_factor: float
    _closed_edges_factor: float

    _merge_priority: bool = False

    @classmethod
    def initialize(cls, attrs):
        cls._degree_dist = attrs.degree_distribution
        cls._degree_trans_probs = attrs.degree_transition_probs
        cls._degree_angles = attrs.degree_angles
        cls._degree_lengths = attrs.degree_lengths
        cls._avg_length = attrs.average_length

        cls._closed_nodes_thr = cls._avg_length * BaseConfig.CLOSED_NODES_FACTOR
        cls._closed_edges_thr = cls._avg_length * BaseConfig.CLOSED_EDGES_FACTOR
        cls._grid_size = cls._avg_length

        cls._closed_nodes_factor = BaseConfig.CLOSED_NODES_FACTOR
        cls._closed_edges_factor = BaseConfig.CLOSED_EDGES_FACTOR

        cls.node_grid = defaultdict(set)
        cls.edge_grid = defaultdict(set)
        cls.reset()

    @classmethod
    def reset(cls):
        cls.id_counter = itertools.count()
        cls.node_grid.clear()
        cls.edge_grid.clear()
        cls._aborted_edge = 0
        cls._merged_edge = 0

    @classmethod
    def create_frozen(cls, position, register_in_grid: bool = False):
        """Create a node that exists for graph construction but won't expand.

        When *register_in_grid* is ``False`` (the default for Phase 2),
        the node is **not** added to ``node_grid``.  Tile edges in
        ``edge_grid`` are sufficient to prevent new growth from crossing
        tile interiors; keeping frozen nodes out of ``node_grid`` avoids
        wasteful back-merges from frontier branches into their own tile.
        """
        node = object.__new__(cls)
        node.id = next(cls.id_counter)
        node.position = position
        node.degree = 2
        node.children = [None, None]
        node.parent = None
        node.clockwise = False
        node.base_angle = 0.0
        if register_in_grid:
            key = cls._spatial_hash(position)
            cls.node_grid[key].add(node)
        return node

    @classmethod
    def create_frontier_node(cls, position, degree, base_angle, clockwise, parent_node):
        """Create a frontier node ready for ``generate_children()``.

        Mimics a normal non-root node whose BFS expansion was paused:
        ``children = [parent_node]`` so ``generate_children()`` will treat
        it as unexpanded (``len(children) == 1``).
        """
        node = object.__new__(cls)
        node.id = next(cls.id_counter)
        node.position = position
        node.degree = degree
        node.base_angle = base_angle
        node.clockwise = clockwise
        node.parent = parent_node
        node.children = [parent_node]
        key = cls._spatial_hash(position)
        cls.node_grid[key].add(node)
        return node

    @classmethod
    def add_frozen_edge(cls, edge):
        """Add an edge to the edge grid without a node context."""
        keys = cls._edge_spatial_hash(*edge)
        for key in keys:
            cls.edge_grid[key].add(edge)

    def __init__(self, position, parent=None, parent_angle=None):
        """
        PRE: param parent and parent_angle must be provided together or not at all.
        POST: The first child of a non-root node is the parent.
        """
        self.id: int = next(GraphNode.id_counter)

        self.position: Tuple[float, float] = position  # position <- (x, y)
        self.clockwise: bool = random.choice([True, False])
        self.parent: GraphNode = parent
        self.children: List[GraphNode] = []
        if parent is not None and parent_angle is not None:
            self.degree = self._choose_degree_by_parent(parent.degree)
            self.base_angle = (parent_angle + 180) % 360
            self._add_child(self.parent)
        elif parent is None and parent_angle is None:
            self.degree = self._choose_degree_random()
            self.base_angle = random.uniform(0, 360)
            self._initialize_root_node()
        else:
            raise ValueError(
                "Parent and parent_angle must be provided together or not at all."
            )

    @staticmethod
    def _choose_degree_by_parent(parent_degree) -> int:
        degrees = list(GraphNode._degree_trans_probs[parent_degree].keys())
        probabilities = list(GraphNode._degree_trans_probs[parent_degree].values())
        return np.random.choice(degrees, p=probabilities)

    @staticmethod
    def _choose_degree_random() -> int:
        degrees = list(GraphNode._degree_dist.keys())
        probabilities = list(GraphNode._degree_dist.values())
        return np.random.choice(degrees, p=probabilities)

    def _initialize_root_node(self) -> None:
        """Initialize a root node by generating its first child."""
        length = random.choice(GraphNode._degree_lengths[self.degree])
        child_position = self._polar_to_cartesian([length], [self.base_angle])[0]
        child = GraphNode(child_position, parent=self, parent_angle=self.base_angle)
        self._add_child(child)
        self._add_to_grid(self.position)
        self._add_to_grid(child.position)
        self._add_edge_to_grid((self.position, child_position))

    def _add_child(self, child) -> None:
        """Add a child to this node."""
        assert (
            len(self.children) < self.degree
        ), f"{self} cannot have more than {self.degree} children."
        self.children.append(child)
        self._add_to_grid(child.position)

    def _add_to_grid(self, position) -> None:
        key = self._spatial_hash(position)
        GraphNode.node_grid[key].add(self)

    def _add_edge_to_grid(
        self, edge: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> None:
        keys = self._edge_spatial_hash(*edge)
        for key in keys:
            GraphNode.edge_grid[key].add(edge)

    def generate_children(self) -> bool:
        """
        :return: Return true iff the node has generated children successfully, vice versa.
        """
        if (
            len(self.children) > 1 or self.degree == 1
        ):  # Skip visited nodes or Endpoint has no other child
            return False

        angles, lengths = self._generate_angles_and_lengths()
        children_positions = self._polar_to_cartesian(lengths, angles)
        assert len(children_positions) == self.degree - 1

        for child_position, angle in zip(children_positions, angles):
            if GraphNode._merge_priority:
                self._place_child_merge_priority(child_position, angle)
            else:
                self._place_child_default(child_position, angle)
        return True

    def _place_child_default(self, child_position, angle) -> None:
        """Original logic: close-edge check first, then merge-node check."""
        if self._any_close_edge(child_position):
            GraphNode._aborted_edge += 1
            return
        close_node = self._get_closest_valid_node(child_position)
        new_edge = (self.position, child_position)
        if close_node is not None:
            new_edge = (self.position, close_node.position)
            if not self._check_intersection(new_edge):
                self._add_child(close_node)
                self._add_edge_to_grid(new_edge)
                GraphNode._merged_edge += 1
            else:
                GraphNode._aborted_edge += 1
        else:
            if not self._check_intersection(new_edge):
                child_node = GraphNode(child_position, parent=self, parent_angle=angle)
                self._add_child(child_node)
                self._add_edge_to_grid(new_edge)
                self._add_to_grid(child_node.position)
            else:
                GraphNode._aborted_edge += 1

    def _place_child_merge_priority(self, child_position, angle) -> None:
        """Scaling mode: check for mergeable nodes first so cross-root
        connections are not blocked by close-edge avoidance."""
        close_node = self._get_closest_valid_node(child_position)
        if close_node is not None:
            new_edge = (self.position, close_node.position)
            if not self._check_intersection(new_edge):
                self._add_child(close_node)
                self._add_edge_to_grid(new_edge)
                GraphNode._merged_edge += 1
            else:
                GraphNode._aborted_edge += 1
            return
        if self._any_close_edge(child_position):
            GraphNode._aborted_edge += 1
            return
        new_edge = (self.position, child_position)
        if not self._check_intersection(new_edge):
            child_node = GraphNode(child_position, parent=self, parent_angle=angle)
            self._add_child(child_node)
            self._add_edge_to_grid(new_edge)
            self._add_to_grid(child_node.position)
        else:
            GraphNode._aborted_edge += 1

    def _get_closest_valid_node(self, position):
        close_nodes_with_distances = self._find_close_node(position)
        if close_nodes_with_distances:
            return min(close_nodes_with_distances, key=lambda x: x[1])[0]
        return None

    def _find_close_node(self, position):
        key = self._spatial_hash(position)
        neighboring_keys = [
            (key[0] + dx, key[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)
        ]
        close_nodes_with_distances = []
        for neighbor_key in neighboring_keys:
            for node in GraphNode.node_grid[neighbor_key]:
                if node != self and node not in self.children:
                    distance = np.linalg.norm(
                        np.array(node.position) - np.array(position)
                    )
                    if distance < self._closed_nodes_thr:
                        close_nodes_with_distances.append((node, distance))
        return close_nodes_with_distances

    def _any_close_edge(self, position) -> bool:
        key = self._spatial_hash(position)
        neighboring_keys = [
            (key[0] + dx, key[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2)
        ]
        for neighbor_key in neighboring_keys:
            for edge in GraphNode.edge_grid[neighbor_key]:
                if self._is_interfering_edge(edge, position):
                    return True
        return False

    def _is_interfering_edge(self, edge, position) -> bool:
        if self.parent and (self.parent.position in edge or self.position in edge):
            return False  # Skip edges from the same parent
        p1, p2 = edge
        return (
            euclidean(position, p1) < self._closed_edges_thr
            or euclidean(position, p2) < self._closed_edges_thr
        )

    def _generate_angles_and_lengths(self) -> Tuple[list[float], ...]:
        if self.degree == 1:
            return [], []
        angles = random.choices(
            GraphNode._degree_angles[self.degree], k=self.degree - 1
        )
        angles = (
            np.cumsum(angles)
            if self.clockwise
            else np.cumsum([-angle for angle in angles])
        )
        angles = (angles + self.base_angle).tolist()
        lengths = random.choices(
            GraphNode._degree_lengths[self.degree], k=self.degree - 1
        )
        return angles, lengths

    def _polar_to_cartesian(
        self, lengths: List[float], angles: List[float]
    ) -> List[Tuple[float, float]]:
        x, y = self.position
        _cartesian_coord = []
        for length, angle in zip(lengths, angles):
            angle_rad = np.deg2rad(angle)
            new_x = x + length * np.cos(angle_rad)
            new_y = y + length * np.sin(angle_rad)
            _cartesian_coord.append((new_x, new_y))
        return _cartesian_coord

    @staticmethod
    def _check_intersection(
        new_edge: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> bool:
        edge_fractions = GraphNode._edge_spatial_hash(*new_edge)
        for edge_frac in edge_fractions:
            for edge in GraphNode.edge_grid[edge_frac]:
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue  # Skip edges with the same endpoint
                if _do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    @staticmethod
    def _spatial_hash(position: Tuple[float, float]) -> Tuple[int, int]:
        grid_size = GraphNode._grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    @staticmethod
    def _edge_spatial_hash(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Set[Tuple[int, int]]:
        grid_size = GraphNode._grid_size
        x_min, x_max = sorted([p1[0], p2[0]])
        y_min, y_max = sorted([p1[1], p2[1]])
        keys = {
            (int(x // grid_size), int(y // grid_size))
            for x in np.arange(x_min, x_max + grid_size, grid_size)
            for y in np.arange(y_min, y_max + grid_size, grid_size)
        }
        return keys

    def __repr__(self):
        return f"GraphNode(id_counter={self.id})"

    def __str__(self):
        return f"Node No. {self.id}"


def _do_intersect(
    p1: Tuple[float, float],
    q1: Tuple[float, float],
    p2: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
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
