"""Read networks off disk in whichever format they were written.

Which formats count as a network is a property of this repo's output, not of
any one run's configuration, so it lives here rather than in a config.  Both
the compare mode and the standalone analysis entry point read the same
directories through this module.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import List

from .csv_io import read_graph_csv
from .synth_graph import SynthGraph

logger = logging.getLogger(__name__)

_EDGELIST_SUFFIX = "_edgelist.csv"
_POSITIONS_SUFFIX = "_positions.csv"


def load_graphs(path: str) -> List[SynthGraph]:
    """Every network at *path*, which may be a directory or a single file.

    A directory is read whole: pickles first, because one holds a batch, then
    the CSV pairs — the original network is only ever written as CSV, so a
    results directory cannot be read without both.
    """
    if os.path.isdir(path):
        graphs = _load_dir(path)
        assert graphs, f"no networks found in {path}"
        return graphs

    assert os.path.isfile(path), f"not a file or directory: {path}"
    return _load_file(path)


def _load_dir(folder: str) -> List[SynthGraph]:
    pickled = _load_pickles(folder)
    if pickled:
        return pickled
    return [
        _load_csv_pair(os.path.join(folder, name))
        for name in sorted(os.listdir(folder))
        if name.endswith(_EDGELIST_SUFFIX)
    ]


def _load_file(path: str) -> List[SynthGraph]:
    if path.endswith(_EDGELIST_SUFFIX):
        return [_load_csv_pair(path)]
    if path.endswith(".pkl"):
        return _from_pickle(_read_pickle(path))
    raise ValueError(
        f"{path}: not a network file. Expected a *{_EDGELIST_SUFFIX} or a .pkl"
    )


def _load_csv_pair(edge_list: str) -> SynthGraph:
    positions = edge_list[: -len(_EDGELIST_SUFFIX)] + _POSITIONS_SUFFIX
    # Named and stopped rather than skipped: half a results directory analysed
    # as if it were whole is a wrong answer, not a smaller one.
    assert os.path.exists(
        positions
    ), f"{edge_list} has no positions file beside it ({positions})"
    return read_graph_csv(edge_list, positions)


def _load_pickles(folder: str) -> List[SynthGraph]:
    for name in sorted(os.listdir(folder)):
        if name.endswith(".pkl") and "network" in name:
            graphs = _from_pickle(_read_pickle(os.path.join(folder, name)))
            if graphs:
                return graphs
    return []


def _read_pickle(path: str):
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if "networkx" in str(exc):
            logger.error(
                f"{path} is a legacy nx.Graph pickle but networkx is not "
                f"installed. pip install networkx"
            )
        raise


def _from_pickle(content) -> List[SynthGraph]:
    """Unwrap a pickle, converting legacy ``nx.Graph`` payloads."""
    if isinstance(content, list):
        return [_as_synth_graph(item) for item in content]
    return [_as_synth_graph(content)]


def _as_synth_graph(obj) -> SynthGraph:
    if isinstance(obj, SynthGraph):
        return obj
    # Only reached for a legacy pickle: unpickling it already required
    # networkx, so importing it here cannot be what fails.
    import networkx as nx

    assert isinstance(obj, nx.Graph), f"not a network: {type(obj).__name__}"
    return SynthGraph.from_networkx(obj)
