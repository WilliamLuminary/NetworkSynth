"""Read and write networks as GraphML, positions included."""

from __future__ import annotations

import igraph as ig
import numpy as np

from .synth_graph import SynthGraph

_GZIP_SUFFIXES = (".graphml.gz", ".graphmlz")
_WEIGHT = "weight"


def write_graph_graphml(graph: SynthGraph, path: str) -> None:
    out = graph.igraph.copy()
    positions = graph.positions()
    out.vs["x"] = positions[:, 0].tolist()
    out.vs["y"] = positions[:, 1].tolist()
    if path.endswith(_GZIP_SUFFIXES):
        out.write_graphmlz(path)
    else:
        out.write_graphml(path)


def read_graph_graphml(path: str) -> SynthGraph:
    if path.endswith(_GZIP_SUFFIXES):
        graph = ig.Graph.Read_GraphMLz(path)
    else:
        graph = ig.Graph.Read_GraphML(path)

    attributes = graph.vs.attributes()
    assert "x" in attributes and "y" in attributes, (
        f"{path} carries no node positions. A network written by this project always "
        "stores them as the 'x' and 'y' vertex attributes."
    )
    positions = np.column_stack(
        (
            np.asarray(graph.vs["x"], dtype=np.float64),
            np.asarray(graph.vs["y"], dtype=np.float64),
        )
    )
    for name in ("x", "y", "id"):
        if name in graph.vs.attributes():
            del graph.vs[name]
    return SynthGraph(graph, positions)
