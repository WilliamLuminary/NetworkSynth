# src/graph/graph_node.py

import random
import numpy as np
from collections import defaultdict


class GraphNode:
    id_counter = 0
    node_grid = defaultdict(set)
    edge_grid = defaultdict(set)
    aborted_edge = 0
    merged_edge = 0

    # Class-level properties
    degree_distribution = {}
    degree_transition_probs = {}
    degree_angles = {}
    degree_edge_lengths = {}
    avg_length = 13  # Default value

    # Parameters
    closed_range = 13  # Default value
    closed_nodes_factor = 1.5
    closed_edges_factor = 1.0
    grid_size = 13  # Default value

    @classmethod
    def reset(cls):
        cls.id_counter = 0
        cls.node_grid.clear()
        cls.edge_grid.clear()
        cls.aborted_edge = 0
        cls.merged_edge = 0

    @classmethod
    def initialize(cls, graph_attributes, config):
        cls.degree_distribution = graph_attributes.degree_distribution
        cls.degree_transition_probs = graph_attributes.degree_transition_probs
        cls.degree_angles = graph_attributes.degree_angles
        cls.degree_edge_lengths = graph_attributes.degree_edge_lengths
        cls.avg_length = graph_attributes.avg_length
        cls.grid_size = graph_attributes.avg_length
        cls.closed_range = graph_attributes.avg_length
        cls.closed_nodes_factor = config.CLOSED_NODES_FACTOR
        cls.closed_edges_factor = config.CLOSED_EDGES_FACTOR

    def __init__(self, position, parent=None, parent_angle=None):
        self.id = GraphNode.id_counter
        GraphNode.id_counter += 1
        self.position = position
        self.parent = parent
        self.children = []
        self.degree = None
        self.base_angle = None
        if parent is not None and parent_angle is not None:
            self.degree = self.choose_degree_based_on_parent(parent.degree)
            self.base_angle = (parent_angle + 180) % 360
            self.add_child(self.parent)
        elif parent is None and parent_angle is None:
            self.degree = self.choose_degree()
            self.base_angle = random.uniform(0, 360)
            self.initialize_root_node()
        else:
            raise ValueError("Parent and parent_angle must be provided together or not at all.")

    def choose_degree(self):
        degrees = list(GraphNode.degree_distribution.keys())
        probabilities = list(GraphNode.degree_distribution.values())
        return np.random.choice(degrees, p=probabilities)

    def choose_degree_based_on_parent(self, parent_degree):
        degrees = list(GraphNode.degree_transition_probs[parent_degree].keys())
        probabilities = list(GraphNode.degree_transition_probs[parent_degree].values())
        return np.random.choice(degrees, p=probabilities)

    def initialize_root_node(self):
        length = random.choice(GraphNode.degree_edge_lengths[self.degree])
        child_position = self.polar_to_cartesian(length, self.base_angle)
        child = GraphNode(child_position, parent=self, parent_angle=self.base_angle)
        self.add_child(child)
        self.add_to_grid(self.position)
        self.add_to_grid(child.position)
        self.add_edge_to_grid((self.position, child_position))

    def add_child(self, child):
        if len(self.children) >= self.degree:
            raise Exception(f"Node {self.id} cannot have more than {self.degree} children.")
        self.children.append(child)
        self.add_to_grid(child.position)

    def add_to_grid(self, position):
        key = self.spatial_hash(position)
        GraphNode.node_grid[key].add(self)

    def add_edge_to_grid(self, edge):
        keys = self.edge_spatial_hash(*edge)
        for key in keys:
            GraphNode.edge_grid[key].add(edge)

    def generate_children(self):
        if len(self.children) > 1 or self.degree == 1:
            return False
        angles, lengths = self.generate_angles_and_lengths()
        for angle, length in zip(angles, lengths):
            child_position = self.polar_to_cartesian(length, angle)
            new_edge = (self.position, child_position)
            if not self.check_intersection(new_edge):
                child_node = GraphNode(child_position, parent=self, parent_angle=angle)
                self.add_child(child_node)
                self.add_edge_to_grid(new_edge)
            else:
                GraphNode.aborted_edge += 1
        return True

    def generate_angles_and_lengths(self):
        if self.degree == 1:
            return [], []
        angles = random.choices(GraphNode.degree_angles[self.degree], k=self.degree - 1)
        angles = np.cumsum(angles)
        angles = (angles + self.base_angle).tolist()
        lengths = random.choices(GraphNode.degree_edge_lengths[self.degree], k=self.degree - 1)
        return angles, lengths

    def polar_to_cartesian(self, length, angle):
        x, y = self.position
        angle_rad = np.deg2rad(angle)
        new_x = x + length * np.cos(angle_rad)
        new_y = y + length * np.sin(angle_rad)
        return (new_x, new_y)

    def check_intersection(self, new_edge):
        keys = self.edge_spatial_hash(*new_edge)
        for key in keys:
            for edge in GraphNode.edge_grid[key]:
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue
                if self.do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    @staticmethod
    def spatial_hash(position):
        grid_size = GraphNode.grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    @staticmethod
    def edge_spatial_hash(p1, p2):
        grid_size = GraphNode.grid_size
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

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        return (o1 != o2 and o3 != o4) or \
            (o1 == 0 and on_segment(p1, p2, q1)) or \
            (o2 == 0 and on_segment(p1, q2, q1)) or \
            (o3 == 0 and on_segment(p2, p1, q2)) or \
            (o4 == 0 and on_segment(p2, q1, q2))
