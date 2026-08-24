import itertools
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from configs import SynthParams


@dataclass(frozen=True)
class TraversalRules:
    """What the BFS compares a candidate placement against.

    Derived, not chosen: the distributions are measured from the original
    network and the thresholds come from those times the run's factors, so this
    holds only what an inner loop reads — squared thresholds and integer cell
    radii, in the form the comparison needs.  What the *caller* chose lives in
    :class:`~configs.params.SynthParams`, which is the single source of the
    factors and the only one of the two that crosses a process boundary.

    Fixed for the whole traversal: nothing here changes as the BFS proceeds.
    """

    degree_dist: Dict[int, float]
    degree_trans_probs: dict
    degree_angles: dict
    degree_lengths: dict

    closed_nodes_thr_sq: float
    closed_edges_thr_sq: float
    #: One grid cell is one mean edge length, so a close-node search only has
    #: to look at the cells within ``closed_nodes_factor`` of the candidate.
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
    """How a traversal's candidate placements turned out.

    A result of the traversal rather than part of it: read it while the
    traversal is open, since leaving the scope clears the record it came from.
    """

    merged: int
    aborted: int

    def __str__(self) -> str:
        return f"merged={self.merged:,}, aborted={self.aborted:,}"


class GraphNode:
    """One node of a BFS traversal, plus the traversal itself.

    Generation records a single BFS rather than placing nodes on a board that
    several callers share, so the record — the id counter, the node and edge
    grids, and the placement counts — lives on the class.  Two consequences
    worth knowing before changing anything here:

    - Exactly one traversal exists per process, so generation parallelises
      across *processes* only, never threads.  Every generation pool in
      ``pipelines/`` is a ``ProcessPoolExecutor`` for this reason; the thread
      pools there render snapshots, which never touch this class.
    - :meth:`traversal` is the scope that owns the record.  The static BFS
      entry points on ``GraphGenerator`` read it straight off the class, so
      they are only callable from inside that block.

    The rules the traversal follows are separate and immutable: see
    :class:`TraversalRules`.
    """

    #: The record: rebuilt by :meth:`reset` for every traversal.
    id_counter = None
    node_grid: Dict[tuple[float, float], set]
    edge_grid: Dict[
        Tuple[int, int], Set[Tuple[Tuple[float, float], Tuple[float, float]]]
    ]
    _aborted_edge: int
    _merged_edge: int

    #: The rules: installed once per traversal, read-only thereafter.
    _rules: Optional[TraversalRules] = None

    _traversal_active = False

    @classmethod
    @contextmanager
    def traversal(cls, attrs, params: SynthParams):
        """Scope one BFS traversal: install its rules, drop its record.

        Leaving the block drops the node and edge grids, so the spatial index
        over a multi-million-node traversal is freed where the traversal ends
        rather than wherever a caller remembers to reset.  Only one may be open
        at a time, for the reason given in the class docstring.
        """
        assert not cls._traversal_active, (
            "a traversal is already open: the record is class-level, so two "
            "cannot be interleaved in one process"
        )
        cls.initialize(attrs, params)
        cls._traversal_active = True
        try:
            yield
        finally:
            cls._traversal_active = False
            cls.reset()

    @classmethod
    def initialize(cls, attrs, params: SynthParams):
        """Install the rules, then start an empty record."""
        cls._rules = TraversalRules.build(attrs, params)
        cls.reset()

    @classmethod
    def reset(cls):
        """Start a fresh record, keeping the rules in place.

        Called at the top of every BFS entry point, so a retry re-draws from
        the same rules rather than reinstalling them.
        """
        cls.id_counter = itertools.count()
        cls.node_grid = defaultdict(set)
        cls.edge_grid = defaultdict(set)
        cls._aborted_edge = 0
        cls._merged_edge = 0

    @classmethod
    def counts(cls) -> PlacementCounts:
        return PlacementCounts(merged=cls._merged_edge, aborted=cls._aborted_edge)

    @classmethod
    def create_interior_node(cls, position):
        """Node for pre-existing tile interior positions.

        Added to ``node_grid`` so Phase 2 frontier expansion can
        discover and merge with existing tile nodes — matching the
        same close-node interaction that Phase 1 BFS has internally.

        Will not expand (``len(children) == 2`` causes
        ``generate_children()`` to return immediately).
        """
        node = object.__new__(cls)
        node.id = next(cls.id_counter)
        node.position = position
        node.degree = 2
        node.children = [None, None]
        node.parent = None
        node.clockwise = False
        node.base_angle = 0.0
        key = cls._spatial_hash(position)
        cls.node_grid[key].add(node)
        return node

    @classmethod
    def create_frontier_node(cls, position, degree, base_angle, clockwise, parent_node):
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
    def register_edge(cls, edge):
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
        transitions = GraphNode._rules.degree_trans_probs[parent_degree]
        degrees = list(transitions.keys())
        probabilities = list(transitions.values())
        return np.random.choice(degrees, p=probabilities)

    @staticmethod
    def _choose_degree_random() -> int:
        degrees = list(GraphNode._rules.degree_dist.keys())
        probabilities = list(GraphNode._rules.degree_dist.values())
        return np.random.choice(degrees, p=probabilities)

    def _initialize_root_node(self) -> None:
        length = random.choice(GraphNode._rules.degree_lengths[self.degree])
        child_position = self._polar_to_cartesian([length], [self.base_angle])[0]
        child = GraphNode(child_position, parent=self, parent_angle=self.base_angle)
        self._add_child(child)
        self._add_to_grid(self.position)
        self._add_to_grid(child.position)
        self._add_edge_to_grid((self.position, child_position))

    def _add_child(self, child) -> None:
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
        if (
            len(self.children) > 1 or self.degree == 1
        ):  # Skip visited nodes or Endpoint has no other child
            return False

        angles, lengths = self._generate_angles_and_lengths()
        children_positions = self._polar_to_cartesian(lengths, angles)
        assert len(children_positions) == self.degree - 1

        for child_position, angle in zip(children_positions, angles):
            self._place_child(child_position, angle)
        return True

    def _place_child(self, child_position, angle) -> None:
        """Place a candidate child node.

        Order of checks (same in Phase 1 and Phase 2):
          1. Close-node merge — if there is an existing node within the
             merge threshold, add an edge to it (counts as a child but
             the merged node is NOT re-queued for expansion).
          2. Close-edge avoidance — if the candidate position is too
             close to an existing edge, abort.
          3. Otherwise create a new child node at the candidate position.
        """
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
        px, py = position
        rules = GraphNode._rules
        thr_sq = rules.closed_nodes_thr_sq
        r = rules.node_search_radius
        close_nodes_with_distances = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for node in GraphNode.node_grid.get((key[0] + dx, key[1] + dy), ()):
                    if node != self and node not in self.children:
                        ddx = node.position[0] - px
                        ddy = node.position[1] - py
                        dist_sq = ddx * ddx + ddy * ddy
                        if dist_sq < thr_sq:
                            close_nodes_with_distances.append((node, dist_sq))
        return close_nodes_with_distances

    def _any_close_edge(self, position) -> bool:
        key = self._spatial_hash(position)
        r = GraphNode._rules.edge_search_radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for edge in GraphNode.edge_grid.get((key[0] + dx, key[1] + dy), ()):
                    if self._is_interfering_edge(edge, position):
                        return True
        return False

    def _is_interfering_edge(self, edge, position) -> bool:
        if self.parent and (self.parent.position in edge or self.position in edge):
            return False
        p1, p2 = edge
        thr_sq = GraphNode._rules.closed_edges_thr_sq
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
        raw = random.choices(
            GraphNode._rules.degree_angles[self.degree], k=self.degree - 1
        )
        sign = 1 if self.clockwise else -1
        base = self.base_angle
        acc = 0.0
        angles = []
        for a in raw:
            acc += sign * a
            angles.append(acc + base)
        lengths = random.choices(
            GraphNode._rules.degree_lengths[self.degree], k=self.degree - 1
        )
        return angles, lengths

    def _polar_to_cartesian(
        self, lengths: List[float], angles: List[float]
    ) -> List[Tuple[float, float]]:
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

    @staticmethod
    def _check_intersection(
        new_edge: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> bool:
        edge_fractions = GraphNode._edge_spatial_hash(*new_edge)
        for edge_frac in edge_fractions:
            for edge in GraphNode.edge_grid.get(edge_frac, ()):
                if new_edge[0] in edge or new_edge[1] in edge:
                    continue  # Skip edges with the same endpoint
                if _do_intersect(new_edge[0], new_edge[1], edge[0], edge[1]):
                    return True
        return False

    @staticmethod
    def _spatial_hash(position: Tuple[float, float]) -> Tuple[int, int]:
        grid_size = GraphNode._rules.grid_size
        return int(position[0] // grid_size), int(position[1] // grid_size)

    @staticmethod
    def _edge_spatial_hash(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Set[Tuple[int, int]]:
        gs = GraphNode._rules.grid_size
        gx1 = int(p1[0] // gs)
        gx2 = int(p2[0] // gs)
        gy1 = int(p1[1] // gs)
        gy2 = int(p2[1] // gs)
        if gx1 > gx2:
            gx1, gx2 = gx2, gx1
        if gy1 > gy2:
            gy1, gy2 = gy2, gy1
        return {(x, y) for x in range(gx1, gx2 + 1) for y in range(gy1, gy2 + 1)}

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
