"""Convert between the two graph libraries, for the duration of the igraph migration.

Temporary by design. It exists so that a consumer still written against networkit keeps
working after `SynthGraph` has moved to igraph internally, letting the migration proceed
one caller at a time. It goes away with the last such consumer.

Node indices are identical in both libraries (0..n-1), so conversion never renumbers.
Weights live on the networkit graph itself and on igraph's ``es["weight"]``.
"""

from __future__ import annotations

import igraph as ig
import networkit as nk


def ig_to_nk(graph: ig.Graph) -> nk.Graph:
    weighted = "weight" in graph.es.attributes()
    out = nk.Graph(graph.vcount(), weighted=weighted)
    if weighted:
        for source, target, weight in zip(
            (e.source for e in graph.es),
            (e.target for e in graph.es),
            graph.es["weight"],
        ):
            out.addEdge(source, target, weight)
    else:
        for edge in graph.es:
            out.addEdge(edge.source, edge.target)
    return out


def nk_to_ig(graph: nk.Graph) -> ig.Graph:
    out = ig.Graph(n=graph.numberOfNodes())
    if graph.isWeighted():
        pairs = []
        weights = []
        for source, target, weight in graph.iterEdgesWeights():
            pairs.append((source, target))
            weights.append(weight)
        out.add_edges(pairs)
        out.es["weight"] = weights
    else:
        out.add_edges(list(graph.iterEdges()))
    return out
