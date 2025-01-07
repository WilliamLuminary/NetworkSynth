# src/original_graph/graph_node.py

import random
from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

from config import Config


class GraphNode:
    id_counter: int = 0
    node_grid: DefaultDict[tuple[float, float], set] = None
    edge_grid: DefaultDict[Tuple[int, int], Set[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    _aborted_edge: int = 0
    _merged_edge: int = 0

    # Class-level properties
    _degree_dist = {}
    _degree_trans_probs = {}
    _degree_angles = {}
    _degree_edge_lengths = {}
    _avg_length: float = 0

    # Class-level Parameters
    _closed_nodes_thr = 0
    _closed_edges_thr = 0
    _closed_nodes_factor = 0
    _closed_edges_factor = 0
    _grid_size = 0

    @classmethod
    def reset(cls):
        cls.id_counter = 0
        cls.node_grid.clear()
        cls.edge_grid.clear()
        cls._aborted_edge = 0
        cls._merged_edge = 0

    @classmethod
    def initialize(cls, graph_attributes):
        cls._degree_dist = graph_attributes.degree_dist
        cls._degree_trans_probs = graph_attributes.degree_trans_probs
        cls._degree_angles = graph_attributes.degree_angles
        cls._degree_edge_lengths = graph_attributes.degree_edge_lengths
        cls._avg_length = graph_attributes.avg_length

        cls._closed_nodes_factor = Config.CLOSED_NODES_FACTOR
        cls._closed_edges_factor = Config.CLOSED_EDGES_FACTOR
        cls._closed_nodes_thr = graph_attributes.avg_length * Config.CLOSED_NODES_FACTOR
        cls._closed_edges_thr = graph_attributes.avg_length * Config.CLOSED_EDGES_FACTOR
        cls._grid_size = graph_attributes.avg_length

        cls.node_grid = defaultdict(set)
        cls.edge_grid = defaultdict(set)

    def __init__(self, position, parent=None, parent_angle=None):
        """
        PRE: param parent and parent_angle must be provided together or not at all.
        POST: The first child of a non-root node is the parent.
        """
        self.id: int = GraphNode.id_counter
        GraphNode.id_counter += 1

        self.position: Tuple[float, float] = position  # position <- (x, y)
        self.clockwise = random.choice([True, False])
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
            raise ValueError("Parent and parent_angle must be provided together or not at all.")

    @staticmethod
    def _choose_degree_random() -> int:
        degrees = list(GraphNode._degree_dist.keys())
        probabilities = list(GraphNode._degree_dist.values())
        return np.random.choice(degrees, p=probabilities)

    @staticmethod
    def _choose_degree_by_parent(parent_degree) -> int:
        degrees = list(GraphNode._degree_trans_probs[parent_degree].keys())
        probabilities = list(GraphNode._degree_trans_probs[parent_degree].values())
        return np.random.choice(degrees, p=probabilities)

    def _initialize_root_node(self) -> None:
        length = random.choice(GraphNode._degree_edge_lengths[self.degree])
        child_position = self._polar_to_cartesian([length], [self.base_angle])[0]
        child = GraphNode(child_position, parent=self, parent_angle=self.base_angle)
        self._add_child(child)
        self._add_to_grid(self.position)
        self._add_to_grid(child.position)
        self._add_edge_to_grid((self.position, child_position))

    def _add_child(self, child) -> None:
        if len(self.children) >= self.degree:
            raise Exception(f"{self} cannot have more than {self.degree} children.")
        self.children.append(child)
        self._add_to_grid(child.position)

    def _add_to_grid(self, position) -> None:
        key = self._spatial_hash(position)
        GraphNode.node_grid[key].add(self)

    def _add_edge_to_grid(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        keys = self._edge_spatial_hash(*edge)
        for key in keys:
            GraphNode.edge_grid[key].add(edge)

    def generate_children(self) -> bool:
        """
        :return: Return true iff the node has generated children successfully, vice versa.
        """
        if len(self.children) > 1 or self.degree == 1:  # Skip visited nodes or Endpoint has no other child
            return False

        angles, lengths = self._generate_angles_and_lengths()
        children_positions = self._polar_to_cartesian(lengths, angles)
        assert len(children_positions) == self.degree - 1

        for child_position, angle in zip(children_positions, angles):
            if self._find_close_edge(child_position):  # Avoid closed edges
                GraphNode._aborted_edge += 1
                continue
            close_node = self._get_closest_valid_node(child_position)
            new_edge = (self.position, child_position)
            if close_node is not None:  # Merge closed nodes
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
        return True

    # def generate_children(self) -> bool:
    #     if len(self.children) > 1 or self.degree == 1:  # Skip visited nodes or Endpoint has no other child
    #         return False
    #     angles, lengths = self._generate_angles_and_lengths()
    #     children_positions = self._polar_to_cartesian(lengths, angles)
    #     assert len(children_positions) == self.degree - 1
    #
    #     for child_position, angle in zip(children_positions, angles):
    #         close_node = self._get_closest_valid_node(child_position)
    #         if close_node is not None:
    #             new_edge = (self.position, close_node.position)
    #             if not self._check_intersection(new_edge):
    #                 self._add_child(close_node)
    #                 self._add_edge_to_grid(new_edge)
    #                 GraphNode._merged_edge += 1
    #             else:
    #                 GraphNode._aborted_edge += 1
    #         else:
    #             if self._find_close_edge(child_position):
    #                 GraphNode._aborted_edge += 1
    #                 continue
    #             new_edge = (self.position, child_position)
    #             if not self._check_intersection(new_edge):
    #                 child_node = GraphNode(child_position, parent=self, parent_angle=angle)
    #                 self._add_child(child_node)
    #                 self._add_edge_to_grid(new_edge)
    #                 self._add_to_grid(child_node.position)
    #             else:
    #                 GraphNode._aborted_edge += 1
    #     return True

    def _get_closest_valid_node(self, position):
        close_nodes_with_distances = self._find_close_node(position)
        if close_nodes_with_distances:
            return min(close_nodes_with_distances, key=lambda x: x[1])[0]
        return None

    def _find_close_node(self, position):
        key = self._spatial_hash(position)
        neighboring_keys = [
            (key[0] + dx, key[1] + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
        ]
        close_nodes_with_distances = []
        for neighbor_key in neighboring_keys:
            for node in GraphNode.node_grid[neighbor_key]:
                if node != self and node not in self.children:
                    distance = np.linalg.norm(np.array(node.position) - np.array(position))
                    if distance < self._closed_nodes_thr:
                        close_nodes_with_distances.append((node, distance))
        return close_nodes_with_distances

    def _find_close_edge(self, position):
        key = self._spatial_hash(position)
        neighboring_keys = [
            (key[0] + dx, key[1] + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
        ]
        for neighbor_key in neighboring_keys:
            for edge in GraphNode.edge_grid[neighbor_key]:
                if self._is_non_sibling_edge(edge, position):
                    return True
        return False

    def _is_non_sibling_edge(self, edge, position):
        if self.parent and (self.parent.position in edge or self.position in edge):
            return False  # Skip edges from the same parent
        p1, p2 = edge
        _distances = [
            euclidean(position, p1),
            euclidean(position, p2)
        ]
        return any(_distance < self._closed_edges_thr for _distance in _distances)

    def _generate_angles_and_lengths(self) -> Tuple[list[float], list[float]]:
        if self.degree == 1:
            return [], []
        angles = random.choices(GraphNode._degree_angles[self.degree], k=self.degree - 1)
        angles = np.cumsum(angles) if self.clockwise else np.cumsum([-angle for angle in angles])
        angles = (angles + self.base_angle).tolist()
        lengths = random.choices(GraphNode._degree_edge_lengths[self.degree], k=self.degree - 1)
        return angles, lengths

    def _polar_to_cartesian(self, lengths: List[float], angles: List[float]) -> List[Tuple[float, float]]:
        x, y = self.position
        _cartesian_coord = []
        for length, angle in zip(lengths, angles):
            angle_rad = np.deg2rad(angle)
            new_x = x + length * np.cos(angle_rad)
            new_y = y + length * np.sin(angle_rad)
            _cartesian_coord.append((new_x, new_y))
        return _cartesian_coord

    @staticmethod
    def _check_intersection(new_edge) -> bool:
        edge_fractions = GraphNode._edge_spatial_hash(*new_edge)
        for edge_frac in edge_fractions:
            for edge in GraphNode.edge_grid[edge_frac]:
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue  # Skip edges with the same endpoint
                if GraphNode._do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    @staticmethod
    def _spatial_hash(position: Tuple[float, float]) -> Tuple[int, int]:
        grid_size = GraphNode._grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    @staticmethod
    def _edge_spatial_hash(p1: Tuple[float, float], p2: Tuple[float, float]) -> Set[Tuple[int, int]]:
        grid_size = GraphNode._grid_size
        x_min, x_max = sorted([p1[0], p2[0]])
        y_min, y_max = sorted([p1[1], p2[1]])
        keys = {(int(x // grid_size), int(y // grid_size))
                for x in np.arange(x_min, x_max + grid_size, grid_size)
                for y in np.arange(y_min, y_max + grid_size, grid_size)}
        return keys

    @staticmethod
    def _do_intersect(p1: Tuple[float, float], q1: Tuple[float, float], p2: Tuple[float, float],
                      q2: Tuple[float, float]) -> bool:
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            return 0 if val == 0 else 1 if val > 0 else 2

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        return ((o1 != o2 and o3 != o4) or
                (o1 == 0 and on_segment(p1, p2, q1)) or
                (o2 == 0 and on_segment(p1, q2, q1)) or
                (o3 == 0 and on_segment(p2, p1, q2)) or
                (o4 == 0 and on_segment(p2, q1, q2)))

    def __repr__(self):
        return f"GraphNode(id_counter={self.id})"

    def __str__(self):
        return f"Node No. {self.id}"
