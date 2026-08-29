import logging
from typing import Dict, List

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from utils import finalize_plot

from .multifractal_processor import MultifractalProcessor

logger = logging.getLogger(__name__)

_PLOT_CONFIG = {
    "font.size": 24,
    "axes.linewidth": 2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}

_CMAP_RANGES = {
    "synthetic": {"name": "Blues", "start": 0.4, "end": 0.8},
    "original": {"name": "Reds", "start": 0.8, "end": 1},
}

_PLOT_STYLE = {
    "synthetic": {"alpha": 0.6, "lw": 3},
    "original": {"alpha": 1.0, "lw": 3},
}


class MultifractalBatchProcessor:

    def __init__(
        self,
        original: List = None,
        synthetic: List = None,
        original_processed: bool = False,
        synthetic_processed: bool = False,
        *,
        measure_weighted: bool,
        full_q_band: bool,
    ):
        self._measure_weighted = measure_weighted
        self._full_q_band = full_q_band
        self._original_data = original
        self._original_processed: bool = original_processed
        self._synthetic_data = synthetic
        self._synthetic_processed: bool = synthetic_processed
        self._images = {}

    @classmethod
    def from_dict(cls, data_: Dict, *, measure_weighted: bool, full_q_band: bool):
        return cls(
            original=data_["original"],
            synthetic=data_["synthetic"],
            measure_weighted=measure_weighted,
            full_q_band=full_q_band,
        )

    def process(self):
        if not self._original_processed:
            logger.info("Processing Multifractal Data.")
            assert self._synthetic_data, "No synthetic data available."
            synthetic_results = self._process_batch(self._synthetic_data)
            assert len(synthetic_results) == len(self._synthetic_data)
            self._synthetic_data = synthetic_results
            self._original_processed = True

        if not self._synthetic_processed:
            assert self._original_data, "No original data available."
            original_results = self._process_batch(self._original_data)
            assert len(original_results) == len(self._original_data)
            self._original_data = original_results
            self._synthetic_processed = True

        return self

    def get_original_data(self):
        assert self._original_processed, "Original data has not been processed."
        return self._original_data

    def get_synthetic_data(self):
        assert self._synthetic_processed, "Synthetic data has not been processed."
        return self._synthetic_data

    def _process_batch(self, graphs):
        processor = MultifractalProcessor(
            graphs, self._measure_weighted, self._full_q_band
        )
        processor.analyze()
        return processor.get_summary_data()

    def plot(self):
        self._images.update(
            {
                "spectra": self._plot_spectra(),
                "dimensions": self._plot_dimensions(),
            }
        )

    def get_images(self) -> Dict[str, Figure]:
        return self._images

    def _plot_spectra(self):
        return self._create_plot(
            "al_list",
            "fal_list",
            r"$\alpha$ (Hölder Exponent)",
            r"$f(\alpha)$ (Multifractal Spectrum)",
        )

    def _plot_dimensions(self):
        return self._create_plot(
            "valid_q",
            "dim_list",
            r"Distorting Exponent $q$",
            r"Generalized Fractal Dimension $D(q)$",
        )

    def _create_plot(self, x_key, y_key, x_label, y_label):
        with matplotlib.rc_context(_PLOT_CONFIG):
            fig = Figure(figsize=(10, 8), dpi=150)
            ax = fig.add_subplot(111)
            legend = []

            for data_type in ["synthetic", "original"]:
                results = getattr(self, f"_{data_type}_data")
                assert results, f"No {data_type} data available."

                self._plot_dataset(
                    ax=ax,
                    entries=results,
                    data_type=data_type,
                    x_key=x_key,
                    y_key=y_key,
                    legend=legend,
                )

            ax.set_xlabel(x_label, fontweight="bold")
            ax.set_ylabel(y_label, fontweight="bold")
            ax.legend(
                handles=legend,
                loc="upper right",
                frameon=False,
            )

        return finalize_plot(fig)

    @staticmethod
    def _plot_dataset(ax, entries, data_type, x_key, y_key, legend):
        cmap_config = _CMAP_RANGES[data_type]
        style = _PLOT_STYLE[data_type]
        cmap = matplotlib.colormaps[cmap_config["name"]]

        color_values = np.linspace(
            cmap_config["start"],
            cmap_config["end"],
            len(entries),
        )

        for idx, entry in enumerate(entries):
            ax.plot(
                entry[x_key],
                entry[y_key],
                color=cmap(color_values[idx]),
                alpha=style["alpha"],
                lw=style["lw"],
            )

        legend.append(
            Line2D(
                [0],
                [0],
                color=cmap(np.mean(color_values)),
                lw=4,
                label=(f"{data_type.capitalize()} " f"(n={len(entries)})"),
            )
        )
