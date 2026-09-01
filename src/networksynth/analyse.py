# SPDX-License-Identifier: GPL-3.0-or-later
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import sys

from networksynth.analysis import MultifractalProcessor
from networksynth.analysis.spectra_plot import plot_dimensions, plot_spectra
from networksynth.configs import save_json
from networksynth.graphs import load_graphs
from networksynth.handlers import configure_console
from networksynth.utils import save_figure_as_webp

logger = logging.getLogger("analyse")

INPUTS = ["data/output/latest_result/sample_1/original"]
OUTPUT_DIR = "data/output/analysis"

MEASURE_WEIGHTED = False

FULL_Q_BAND = True

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_ARGS = 2
EXIT_INTERRUPTED = 130

_USAGE = """usage: python analyse.py [<input> ...  <output_dir>]

With no arguments, uses the INPUTS and OUTPUT_DIR constants in this script.
With arguments, the last is the output directory and the rest are inputs.

An input is a directory of networks, a *_edgelist.csv, or a .pkl batch.
"""


def _labels_for(paths) -> list:
    stems = []
    for path in paths:
        parts = [p for p in os.path.normpath(path).split(os.sep) if p]
        name = parts[-1]
        for suffix in ("_edgelist.csv", ".pkl"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        stems.append(parts[:-1] + [name])

    for depth in range(1, max(len(stem) for stem in stems) + 1):
        labels = ["_".join(stem[-depth:]) for stem in stems]
        if len(set(labels)) == len(labels):
            return labels

    raise ValueError(f"cannot tell these inputs apart: {list(paths)}")


def analyse(inputs, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for label, path in zip(_labels_for(inputs), inputs):
        graphs = load_graphs(path)
        logger.info(f"{label}: {len(graphs)} network(s) from {path}")

        processor = MultifractalProcessor(graphs, MEASURE_WEIGHTED, FULL_Q_BAND)
        processor.analyze()
        results[label] = processor.get_summary_data()

    data_path = os.path.join(output_dir, "analysis_data.json")
    save_json(
        {
            "results": results,
            "measure_weighted": MEASURE_WEIGHTED,
            "full_q_band": FULL_Q_BAND,
        },
        data_path,
    )
    logger.info(f"Wrote {data_path}")

    for name, figure in (
        ("spectra", plot_spectra(results)),
        ("dimensions", plot_dimensions(results)),
    ):
        figure_path = os.path.join(output_dir, f"analysis_{name}.webp")
        save_figure_as_webp(figure, figure_path)
        logger.info(f"Wrote {figure_path}")


def main(argv=None) -> int:
    configure_console()
    argv = sys.argv if argv is None else argv

    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(_USAGE)
        return EXIT_BAD_ARGS

    if len(argv) == 1:
        inputs, output_dir = INPUTS, OUTPUT_DIR
    elif len(argv) >= 3:
        inputs, output_dir = argv[1:-1], argv[-1]
    else:
        print(_USAGE)
        return EXIT_BAD_ARGS

    logger.info(
        f"Analysing {len(inputs)} set(s) -> {output_dir} "
        f"(measure_weighted={MEASURE_WEIGHTED}, full_q_band={FULL_Q_BAND})"
    )
    try:
        analyse(inputs, output_dir)
    except KeyboardInterrupt:
        logger.critical("Interrupted — exiting.")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("Analysis failed")
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
