import random
from collections import defaultdict
from typing import Dict

import numpy as np


class GraphNode:
    # Must provide
    degree_distribution = {}
    degree_transition_probs = {}
    degree_angles = {}
    degree_edge_lengths = {}
    grid_size = 13  # The default value is the average value of lengths in the original network
    # Manager
    node_grid = defaultdict(set)
    edge_grid = defaultdict(set)
    # Params
    closed_range = 13  # The default value is the average value of lengths in the original network
    closed_nodes_factor = 1.5
    closed_edges_factor = 1
    # Counter
    aborted_edge = 0
    merged_edge = 0
    id_counter = 0
    # Features
    enable_check_intersection = True
    enable_merge_nodes = True
    enable_clockwise = True
    enable_degree_transition_probs = True
    enable_merge_edges = True

    def __init__(self, position, *, parent=None, parent_angle=None):
        """
        PRE: param parent and parent_angle must be provided together or not at all.
        POST: The first child of a non-root node is the parent.
        """
        self.id = GraphNode.id_counter
        GraphNode.id_counter += 1
        self.position = position
        self.clockwise = random.choice([True, False]) if GraphNode.enable_clockwise else True
        self.parent = parent
        self.children = []

        if parent is not None and parent_angle is not None:  # Non-root Node
            self.degree = self.choose_degree_based_on_parent(
                parent.degree) if GraphNode.enable_degree_transition_probs else np.random.choice(
                list(GraphNode.degree_distribution.keys()), p=list(GraphNode.degree_distribution.values()))
            self.base_angle = (parent_angle + 180) % 360
            self.add_child(self.parent)
        elif parent is None and parent_angle is None:  # Root Node
            self.degree = np.random.choice(list(GraphNode.degree_distribution.keys()),
                                           p=list(GraphNode.degree_distribution.values()))
            self.base_angle = random.uniform(0, 360)
            self.initialize_root_node()
        else:
            raise ValueError("Both parent and parent_angle must be provided together or not at all.")

    @staticmethod
    def choose_degree_based_on_parent(parent_degree):
        degrees = list(GraphNode.degree_transition_probs[parent_degree].keys())
        probabilities = list(GraphNode.degree_transition_probs[parent_degree].values())
        return np.random.choice(degrees, p=probabilities)

    def initialize_root_node(self):
        length = random.choice(GraphNode.degree_edge_lengths[self.degree])
        child_position = self.polar_to_cartesian([length], [self.base_angle])[0]
        child = GraphNode(child_position, parent=self, parent_angle=self.base_angle)
        edge = (self.position, child_position)
        self.add_child(child)
        self.add_to_grid(self.position)
        self.add_to_grid(child.position)
        self.add_edge_to_grid(edge)

    def add_child(self, child):
        if len(self.children) >= self.degree:
            raise Exception(
                f"Node {self} cannot have more than {self.degree} children, current children {self.children}")
        self.children.append(child)
        self.add_to_grid(child.position)

    def add_to_grid(self, position):
        key = self.spatial_hash(position)
        GraphNode.node_grid[key].add(self)
        if len(GraphNode.node_grid[key]) > 1000:
            print("Unusual length of nodes collision chain")

    def add_edge_to_grid(self, edge):
        keys = self.edge_spatial_hash(*edge)
        for key in keys:
            GraphNode.edge_grid[key].add(edge)
            if len(GraphNode.edge_grid[key]) > 1000:
                print("Unusual length of edges collision chain")

    # def generate_children(self) -> bool:
    #     if len(self.children) > 1 or self.degree == 1: # Skip visited nodes or Endpoint has no other child
    #         return False
    #     angles, lengths = self.generate_angles_and_lengths()
    #     children_positions = self.polar_to_cartesian(lengths, angles)
    #     assert len(children_positions) == self.degree - 1

    #     for child_position, angle in zip(children_positions, angles):
    #         close_node = self.get_closest_valid_node(child_position)
    #         if GraphNode.enable_merge_nodes and close_node is not None:
    #             new_edge = (self.position, close_node.position)
    #             if not self.check_intersection(new_edge):
    #                 self.add_child(close_node)
    #                 self.add_edge_to_grid(new_edge)
    #                 GraphNode.merged_edge += 1
    #             else:
    #                 GraphNode.aborted_edge += 1
    #         else:
    #             if GraphNode.enable_merge_edges and self.find_close_edge(child_position):
    #                 GraphNode.aborted_edge += 1
    #                 continue
    #             new_edge = (self.position, child_position)
    #             if not self.check_intersection(new_edge):
    #                 child_node = GraphNode(child_position, parent=self, parent_angle=angle)
    #                 self.add_child(child_node)
    #                 self.add_edge_to_grid(new_edge)
    #                 self.add_to_grid(child_node.position)
    #             else:
    #                 GraphNode.aborted_edge += 1
    #     return True

    def generate_children(self) -> bool:
        if len(self.children) > 1 or self.degree == 1:  # Skip visited nodes or Endpoint has no other child
            return False
        angles, lengths = self.generate_angles_and_lengths()
        children_positions = self.polar_to_cartesian(lengths, angles)
        assert len(children_positions) == self.degree - 1

        for child_position, angle in zip(children_positions, angles):
            if GraphNode.enable_merge_edges and self.find_close_edge(child_position):
                GraphNode.aborted_edge += 1
                continue
            close_node = self.get_closest_valid_node(child_position)
            new_edge = (self.position, child_position)
            if GraphNode.enable_merge_nodes and close_node is not None:
                new_edge = (self.position, close_node.position)
                if not self.check_intersection(new_edge):
                    self.add_child(close_node)
                    self.add_edge_to_grid(new_edge)
                    GraphNode.merged_edge += 1
                else:
                    GraphNode.aborted_edge += 1
            else:
                if not self.check_intersection(new_edge):
                    child_node = GraphNode(child_position, parent=self, parent_angle=angle)
                    self.add_child(child_node)
                    self.add_edge_to_grid(new_edge)
                    self.add_to_grid(child_node.position)
                else:
                    GraphNode.aborted_edge += 1
        return True

    def get_closest_valid_node(self, position):
        close_nodes_with_distances = self.find_close_node(position)
        if close_nodes_with_distances:
            return min(close_nodes_with_distances, key=lambda x: x[1])[0]
        return None

    def find_close_node(self, position):
        # threshold = calculate_distance(self.position, position) * GraphNode.closed_nodes_factor
        threshold = GraphNode.closed_range * GraphNode.closed_nodes_factor
        key = self.spatial_hash(position)
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
                    if distance < threshold:
                        close_nodes_with_distances.append((node, distance))
        return close_nodes_with_distances

    def generate_angles_and_lengths(self):
        if self.degree == 1:
            return [], []

        angles = random.choices(list(GraphNode.degree_angles[self.degree]), k=self.degree - 1)
        angles = np.cumsum(angles) if self.clockwise else np.cumsum([-angle for angle in angles])
        angles = (angles + self.base_angle).tolist()

        lengths = random.choices(list(GraphNode.degree_edge_lengths[self.degree]), k=self.degree - 1)

        return angles, lengths

    def polar_to_cartesian(self, lengths, angles):
        x, y = self.position
        cartesian_coords = []
        for length, angle in zip(lengths, angles):
            angle_rad = np.deg2rad(angle)
            new_x = x + length * np.cos(angle_rad)
            new_y = y + length * np.sin(angle_rad)
            cartesian_coords.append((new_x, new_y))
        return cartesian_coords

    def find_close_edge(self, position):
        key = self.spatial_hash(position)
        neighboring_keys = [
            (key[0] + dx, key[1] + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
        ]
        for neighbor_key in neighboring_keys:
            for edge in GraphNode.edge_grid[neighbor_key]:
                if self.is_non_sibling_edge(edge, position):
                    return True
        return False

    def is_non_sibling_edge(self, edge, position):
        # threshold = calculate_distance(self.position, position) * GraphNode.closed_edges_factor
        threshold = GraphNode.closed_range * GraphNode.closed_edges_factor
        if self.parent and (self.parent.position in edge or self.position in edge):
            return False  # Skip edges from the same parent
        p1, p2 = edge
        distances = [
            np.linalg.norm(np.array(position) - np.array(p1)),
            np.linalg.norm(np.array(position) - np.array(p2))
        ]
        return any(distance < threshold for distance in distances)

    def check_intersection(self, new_edge):
        if not GraphNode.enable_check_intersection:
            return False
        keys = self.edge_spatial_hash(*new_edge)
        for key in keys:
            for edge in GraphNode.edge_grid[key]:
                # if new_edge[0] == edge[0] or new_edge[0] == edge[1] or new_edge[1] == edge[0] or new_edge[1] == edge[1]:
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue  # Skip edges with the same endpoint
                if GraphNode.do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    @staticmethod
    def spatial_hash(position, grid_size=None):
        grid_size = grid_size or GraphNode.grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    @staticmethod
    def edge_spatial_hash(p1, p2, grid_size=None):
        grid_size = grid_size or GraphNode.grid_size
        x_min, x_max = sorted([p1[0], p2[0]])
        y_min, y_max = sorted([p1[1], p2[1]])
        keys = {(int(x // grid_size), int(y // grid_size))
                for x in np.arange(x_min, x_max + grid_size, grid_size)
                for y in np.arange(y_min, y_max + grid_size, grid_size)}
        return keys

    @staticmethod
    def do_intersect(p1, q1, p2, q2):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            return 0 if val == 0 else 1 if val > 0 else 2

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1, o2, o3, o4 = orientation(p1, q1, p2), orientation(p1, q1, q2), orientation(p2, q2, p1), orientation(p2, q2,
                                                                                                                q1)
        return (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment(p1, p2, q1)) or (
                o2 == 0 and on_segment(p1, q2, q1)) or (o3 == 0 and on_segment(p2, p1, q2)) or (
                o4 == 0 and on_segment(p2, q1, q2))

    @staticmethod
    def print_attributes():
        attrs_to_print = [
            'degree_distribution',
            'degree_transition_probs',
            'degree_angles',
            'degree_edge_lengths',
            'grid_size'
        ]

        for attr in attrs_to_print:
            value = getattr(GraphNode, attr, None)
            if value is None:
                print(f"{attr}: Attribute not found.")
            elif isinstance(value, (list, dict)):
                print(f"{attr}:")
                if isinstance(value, list):
                    for item in value:
                        print(f"  - {item}")
                elif isinstance(value, dict):
                    for key, val in value.items():
                        print(f"  {key}: {val}")
            else:
                print(f"{attr}: {value}")

    @staticmethod
    def reset():
        GraphNode.node_grid.clear()
        GraphNode.edge_grid.clear()
        GraphNode.aborted_edge = 0
        GraphNode.merged_edge = 0
        GraphNode.id_counter = 0

    def __repr__(self):
        return f"Node (Id. {self.id})"

    def __str__(self):
        return f"Node (Id. {self.id})"

    # Since we use the coordinates of nodes to deal with hashing, no need to overwrite __eq__ and __hash__


def set_graph_node_attribute(**kwargs):
    for attr, value in kwargs.items():
        if hasattr(GraphNode, attr):
            setattr(GraphNode, attr, value)
    GraphNode.reset()


def init_graph_node(degree_distribution: Dict, degree_transition_probs, degree_angles, degree_edge_lengths, **kwargs):
    GraphNode.degree_distribution = degree_distribution.copy()
    GraphNode.degree_transition_probs = degree_transition_probs.copy()
    GraphNode.degree_angles = degree_angles.copy()
    GraphNode.degree_edge_lengths = degree_edge_lengths.copy()

    grid_size = determine_grid_size(
        degree_edge_lengths=degree_edge_lengths, **kwargs
    )
    kwargs['grid_size'] = grid_size if 'grid_size' not in kwargs else kwargs['grid_size']
    set_graph_node_attribute(**kwargs)


def reset_graph_node(if_init=False, **kwargs):
    if if_init:
        init_graph_node(
            degree_distribution=kwargs.pop('degree_distribution', None),
            degree_transition_probs=kwargs.pop('degree_transition_probs', None),
            degree_angles=kwargs.pop('degree_angles', None),
            degree_edge_lengths=kwargs.pop('degree_edge_lengths', None),
            **kwargs
        )
    else:
        set_graph_node_attribute(**kwargs)


def round_to_int(value) -> int:
    return int(round(value, 0))


def determine_grid_size(**kwargs):
    if 'average_length' in kwargs:
        grid_size = kwargs['average_length']
    elif 'degree_edge_lengths' in kwargs:
        all_edge_lengths = [length for lengths in kwargs['degree_edge_lengths'].values() for length in lengths]
        grid_size = np.mean(
            all_edge_lengths) if all_edge_lengths else GraphNode.grid_size  # Default to GraphNode's grid size if empty
    else:
        grid_size = GraphNode.grid_size  # Use the existing default grid size if no length info is provided
    return round_to_int(kwargs.get('regen_factor', 1) * grid_size)
