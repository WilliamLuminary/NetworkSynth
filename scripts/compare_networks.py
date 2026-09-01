# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot an original network and its synthetic counterparts on one figure.

    python scripts/compare_networks.py
    python scripts/compare_networks.py <original> <synthetic> <out_dir>

Both inputs may be a directory of networks, a single ``*_edgelist.csv``, or a
``.pkl`` batch.  For a general analysis of any number of sets, use ``analyse.py``
at the repo root; this script exists for the specific original-vs-synthetic
figure, which is why the two sets keep their established colours.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging

from analysis import MultifractalProcessor
from analysis.spectra_plot import plot_dimensions, plot_spectra
from configs import save_json
from graphs import load_graphs
from utils import save_figure_as_webp

logger = logging.getLogger("compare_networks")

_RESULTS = "data/output/latest_result/sample_1"
ORIGINAL = os.path.join(_RESULTS, "original")
SYNTHETIC = os.path.join(_RESULTS, "synthetic")
OUTPUT_DIR = "data/output/comparison"

#: Measure edge widths, or topology only.  False puts a weighted original and an
#: unweighted synthetic on the same footing.
MEASURE_WEIGHTED = False

#: Wide q band (401 points) instead of the narrow one (61).
FULL_Q_BAND = True

#: None = half the cores.  Analysis of each network is independent.
MAX_WORKERS = None

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_ARGS = 2
EXIT_INTERRUPTED = 130

_USAGE = """usage: python scripts/compare_networks.py [<original> <synthetic> <out_dir>]

With no arguments, uses the constants at the top of this script.
"""


def compare(original_path: str, synthetic_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # "original" and "synthetic" are the labels the spectra plot styles by name,
    # so this figure keeps the colours it has always had.
    results = {}
    for label, path in (("original", original_path), ("synthetic", synthetic_path)):
        graphs = load_graphs(path)
        logger.info(f"{label}: {len(graphs)} network(s) from {path}")

        processor = MultifractalProcessor(
            graphs, MEASURE_WEIGHTED, FULL_Q_BAND, max_workers=MAX_WORKERS
        )
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
        figure_path = os.path.join(output_dir, f"comparison_{name}.webp")
        save_figure_as_webp(figure, figure_path)
        logger.info(f"Wrote {figure_path}")


def main(argv=None) -> int:
    from handlers import configure_console

    configure_console()
    argv = sys.argv if argv is None else argv

    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print(_USAGE)
        return EXIT_BAD_ARGS

    if len(argv) == 1:
        original, synthetic, output_dir = ORIGINAL, SYNTHETIC, OUTPUT_DIR
    elif len(argv) == 4:
        original, synthetic, output_dir = argv[1], argv[2], argv[3]
    else:
        print(_USAGE)
        return EXIT_BAD_ARGS

    logger.info(
        f"Comparing {synthetic} against {original} -> {output_dir} "
        f"(measure_weighted={MEASURE_WEIGHTED}, full_q_band={FULL_Q_BAND})"
    )
    try:
        compare(original, synthetic, output_dir)
    except KeyboardInterrupt:
        logger.critical("Interrupted — exiting.")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("Comparison failed")
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
