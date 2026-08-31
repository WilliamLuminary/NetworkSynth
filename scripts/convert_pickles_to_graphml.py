"""One-off: rewrite pickled SynthGraph files as GraphML.

Those pickles embed a networkit graph, so they stop being readable the moment the
project moves off networkit -- which is the argument against the format. Run this
once, while networkit is still installed, then delete this script.

    python scripts/convert_pickles_to_graphml.py <file-or-dir> [more...]
"""

import os
import pickle
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from graphs.backend import nk_to_ig
from graphs.graphml_io import write_graph_graphml
from graphs.synth_graph import SynthGraph


def _as_synth_graph(obj) -> SynthGraph:
    """Rebuild on igraph, whichever backend the pickle happens to carry."""
    graph, positions = obj._graph, obj._positions
    if type(graph).__module__.startswith("networkit"):
        graph = nk_to_ig(graph)
    return SynthGraph(graph, np.asarray(positions, dtype=np.float64))


def convert(path: str) -> None:
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)

    items = loaded if isinstance(loaded, list) else [loaded]
    stem = path[: -len(".pkl")] if path.endswith(".pkl") else path
    for index, item in enumerate(items):
        graph = _as_synth_graph(item)
        suffix = "" if len(items) == 1 else f"_n{index}"
        out = f"{stem}{suffix}.graphml"
        write_graph_graphml(graph, out)
        print(
            f"  {path} -> {out}  "
            f"(nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
            f"weighted={graph.is_weighted()})"
        )


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("usage: convert_pickles_to_graphml.py <file-or-dir> [more...]")
    for target in argv[1:]:
        if os.path.isdir(target):
            for name in sorted(os.listdir(target)):
                if name.endswith(".pkl"):
                    convert(os.path.join(target, name))
        else:
            convert(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
