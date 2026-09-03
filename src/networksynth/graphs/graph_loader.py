# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
import os
from typing import List

from .csv_io import read_graph_csv
from .synth_graph import SynthGraph

logger = logging.getLogger(__name__)

_EDGELIST_SUFFIX = "_edgelist.csv"
_POSITIONS_SUFFIX = "_positions.csv"
# StructuralGT's exporter names the same two files differently.
_SGT_EDGELIST_SUFFIX = "_EdgeList.csv"
_SGT_POSITIONS_SUFFIX = "_NodePositions.csv"
_CSV_PAIRS = (
    (_EDGELIST_SUFFIX, _POSITIONS_SUFFIX),
    (_SGT_EDGELIST_SUFFIX, _SGT_POSITIONS_SUFFIX),
)
_EDGELIST_SUFFIXES = tuple(edges for edges, _ in _CSV_PAIRS)
_GRAPHML_SUFFIXES = (".graphml", ".graphml.gz", ".graphmlz")


def load_graphs(path: str) -> List[SynthGraph]:
    if os.path.isdir(path):
        graphs = _load_dir(path)
        assert graphs, f"no networks found in {path}"
        return graphs

    assert os.path.isfile(path), f"not a file or directory: {path}"
    return _load_file(path)


def _load_dir(folder: str) -> List[SynthGraph]:
    graphml = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.endswith(_GRAPHML_SUFFIXES)
    ]
    if graphml:
        from .graphml_io import read_graph_graphml

        return [read_graph_graphml(path) for path in graphml]

    return [
        _load_csv_pair(os.path.join(folder, name))
        for name in sorted(os.listdir(folder))
        if name.endswith(_EDGELIST_SUFFIXES)
    ]


def _load_file(path: str) -> List[SynthGraph]:
    if path.endswith(_EDGELIST_SUFFIXES):
        return [_load_csv_pair(path)]
    if path.endswith(_GRAPHML_SUFFIXES):
        from .graphml_io import read_graph_graphml

        return [read_graph_graphml(path)]
    raise ValueError(
        f"{path}: not a network file. Expected a *{_EDGELIST_SUFFIX} " f"or a *.graphml"
    )


def _load_csv_pair(edge_list: str) -> SynthGraph:
    edges_suffix, positions_suffix = next(
        pair for pair in _CSV_PAIRS if edge_list.endswith(pair[0])
    )
    positions = edge_list[: -len(edges_suffix)] + positions_suffix
    assert os.path.exists(
        positions
    ), f"{edge_list} has no positions file beside it ({positions})"

    graph = read_graph_csv(edge_list, positions)
    if positions_suffix == _SGT_POSITIONS_SUFFIX:
        # StructuralGT writes the skeleton's (row, col) under headers x and y.
        from networksynth.utils import transpose_positions

        transpose_positions(graph)
    return graph
