# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, List, Mapping

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from utils import finalize_plot

logger = logging.getLogger(__name__)

_PLOT_CONFIG = {
    "font.size": 24,
    "axes.linewidth": 2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


@dataclass(frozen=True)
class _Style:
    cmap: str
    start: float
    end: float
    alpha: float
    lw: float = 3


_STYLES = {
    "synthetic": _Style("Blues", 0.4, 0.8, alpha=0.6),
    "original": _Style("Reds", 0.8, 1.0, alpha=1.0),
}

_SPARE_STYLES = (
    _Style("Greens", 0.4, 0.8, alpha=0.8),
    _Style("Purples", 0.4, 0.8, alpha=0.8),
    _Style("Oranges", 0.4, 0.8, alpha=0.8),
)


def plot_spectra(datasets: Mapping[str, List[Dict]]) -> Figure:
    return _create_plot(
        datasets,
        "al_list",
        "fal_list",
        r"$\alpha$ (Hölder Exponent)",
        r"$f(\alpha)$ (Multifractal Spectrum)",
    )


def plot_dimensions(datasets: Mapping[str, List[Dict]]) -> Figure:
    return _create_plot(
        datasets,
        "valid_q",
        "dim_list",
        r"Distorting Exponent $q$",
        r"Generalized Fractal Dimension $D(q)$",
    )


def _titled(label: str) -> str:
    text = label.replace("_", " ")
    return text[:1].upper() + text[1:]


def _styles_for(labels) -> Dict[str, _Style]:
    spare = cycle(_SPARE_STYLES)
    return {
        label: _STYLES[label] if label in _STYLES else next(spare) for label in labels
    }


def _create_plot(
    datasets: Mapping[str, List[Dict]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> Figure:
    assert datasets, "nothing to plot: no labelled result sets given"
    for label, results in datasets.items():
        assert results, f"no results for {label!r}"

    styles = _styles_for(datasets)
    with matplotlib.rc_context(_PLOT_CONFIG):
        fig = Figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111)
        legend = []

        for label, results in datasets.items():
            _plot_dataset(ax, results, label, styles[label], x_key, y_key, legend)

        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel(y_label, fontweight="bold")
        ax.legend(handles=legend, loc="upper right", frameon=False)

    return finalize_plot(fig)


def _plot_dataset(ax, entries, label, style, x_key, y_key, legend) -> None:
    cmap = matplotlib.colormaps[style.cmap]
    color_values = np.linspace(style.start, style.end, len(entries))

    for idx, entry in enumerate(entries):
        ax.plot(
            entry[x_key],
            entry[y_key],
            color=cmap(color_values[idx]),
            alpha=style.alpha,
            lw=style.lw,
        )

    legend.append(
        Line2D(
            [0],
            [0],
            color=cmap(np.mean(color_values)),
            lw=4,
            label=f"{_titled(label)} (n={len(entries)})",
        )
    )
