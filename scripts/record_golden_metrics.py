"""Record every analyzer output, with timings, as an oracle to diff against.

    python scripts/record_golden_metrics.py [out_dir]
"""

import json
import os
import sys
import time
from dataclasses import astuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.multifractal_analyzer import MultifractalAnalyzer
from graphs.graph_loader import load_graphs

SAMPLES = {
    "sample_1": "data/input/samples/gui_mode/sample_1_edgelist.csv",
    "sample_2": "data/input/samples/gui_mode/sample_2_edgelist.csv",
    "sample_3": "data/input/samples/gui_mode/sample_3_edgelist.csv",
}
DEFAULT_OUT = os.path.join("tests", "data", "golden_metrics")


def _plain(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _timed(label, timings, fn):
    start = time.perf_counter()
    result = fn()
    timings[label] = round(time.perf_counter() - start, 6)
    return result


def record(name: str, path: str) -> dict:
    graph = load_graphs(path)[0]
    entry = {
        "source": path,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weighted": graph.is_weighted(),
        "runs": {},
    }
    for measure_weighted in (True, False):
        timings: dict = {}
        analyzer = MultifractalAnalyzer(graph, measure_weighted, full_q_band=True)
        features = _timed(
            "analyze_error_features", timings, analyzer.analyze_error_features
        )
        analysis = _timed("analyze_graph", timings, analyzer.analyze_graph)
        entry["runs"][f"measure_weighted={measure_weighted}"] = {
            "error_features": list(astuple(features)),
            "analysis": analysis,
            "seconds": timings,
        }
        print(
            f"  {name} weighted={measure_weighted}: "
            f"features={astuple(features)} "
            f"({timings['analyze_error_features']:.2f}s + "
            f"{timings['analyze_graph']:.2f}s)"
        )
    return entry


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    out_dir = argv[1] if len(argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    for name, path in SAMPLES.items():
        entry = record(name, path)
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w") as handle:
            json.dump(entry, handle, indent=2, default=_plain, sort_keys=True)
        print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
