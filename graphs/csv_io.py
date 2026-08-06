# src/graphs/csv_io.py
"""Read a graph from an edge-list CSV plus a node-positions CSV.

This is the input half of the StructuralGT file contract: they already know how
to write these two files via their existing ``export_edge_list`` and
``export_node_positions`` options, so nothing new is needed on their side.

Two header conventions are accepted, since both appear in practice:

* StructuralGT — ``Source,Target`` (plus ``Weight,Length,Width,Angle`` when
  weighted), and ``x,y`` for positions.
* NetworkSynth's own :func:`configs.file_definitions.save_network_csv` —
  ``source_index,target_index,edge_weight`` and ``x,y``.

Node indices are row numbers into the positions file.  That alignment is an
assumption about the writer, not something the files state, so it is validated
rather than trusted: an out-of-range index raises instead of producing a graph
with silently wrong geometry.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

import numpy as np

from .synth_graph import SynthGraph

_SOURCE_NAMES = ("source", "source_index", "from", "u")
_TARGET_NAMES = ("target", "target_index", "to", "v")
_WEIGHT_NAMES = ("weight", "edge_weight")
_X_NAMES = ("x", "pos_x")
_Y_NAMES = ("y", "pos_y")


def _column(header: List[str], candidates: Tuple[str, ...], path: str) -> int:
    lowered = [h.strip().lower() for h in header]
    for name in candidates:
        if name in lowered:
            return lowered.index(name)
    raise ValueError(f"{path}: no column matching {candidates!r}; found {header!r}")


def _optional_column(header: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
    lowered = [h.strip().lower() for h in header]
    for name in candidates:
        if name in lowered:
            return lowered.index(name)
    return None


def _read_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="") as handle:
        rows = [r for r in csv.reader(handle) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: file is empty")
    return rows[0], rows[1:]


def read_positions_csv(path: str) -> np.ndarray:
    """Read an ``(N, 2)`` position array from a CSV with ``x`` and ``y`` columns."""
    header, rows = _read_rows(path)
    xi = _column(header, _X_NAMES, path)
    yi = _column(header, _Y_NAMES, path)
    if not rows:
        raise ValueError(f"{path}: header only, no position rows")
    try:
        return np.array([(float(r[xi]), float(r[yi])) for r in rows], dtype=np.float64)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: malformed position row ({exc})") from exc


def read_edge_list_csv(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read an ``(E, 2)`` edge array and, when present, an ``(E,)`` weight array."""
    header, rows = _read_rows(path)
    si = _column(header, _SOURCE_NAMES, path)
    ti = _column(header, _TARGET_NAMES, path)
    wi = _optional_column(header, _WEIGHT_NAMES)

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for line_no, row in enumerate(rows, start=2):
        try:
            edges.append((int(float(row[si])), int(float(row[ti]))))
            if wi is not None:
                weights.append(float(row[wi]))
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"{path}: malformed edge on line {line_no} ({exc})"
            ) from exc

    edge_arr = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)
    return edge_arr, (np.array(weights, dtype=np.float64) if wi is not None else None)


def read_graph_csv(edge_list_path: str, positions_path: str) -> SynthGraph:
    """Build a :class:`SynthGraph` from an edge-list CSV and a positions CSV.

    Raises rather than guessing: an edge referencing a node with no position
    would otherwise yield a graph whose geometry is quietly wrong.
    """
    positions = read_positions_csv(positions_path)
    edges, weights = read_edge_list_csv(edge_list_path)

    if len(edges):
        highest = int(edges.max())
        if highest >= len(positions):
            raise ValueError(
                f"edge list references node index {highest} but "
                f"{positions_path} has only {len(positions)} rows. "
                "Node indices must be row numbers into the positions file; "
                f"check that {edge_list_path} and {positions_path} come from "
                "the same export."
            )
        if int(edges.min()) < 0:
            raise ValueError(f"{edge_list_path}: negative node index")

    graph = SynthGraph.from_edge_list(positions, edges)
    if weights is not None:
        graph.make_weighted()
        for (u, v), weight in zip(edges, weights):
            graph.set_weight(int(u), int(v), float(weight))
    return graph
