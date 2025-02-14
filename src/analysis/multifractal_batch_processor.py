from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from analysis import MultifractalProcessor
from utils.utils import finalize_plot


class MultifractalBatchProcessor:
    # noinspection SpellCheckingInspection
    _PLOT_CONFIG = {
        'font.size': 24,
        'axes.linewidth': 2,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True
    }

    _CMAP_RANGES = {
        'synthetic': {'name': 'Blues', 'start': 0.4, 'end': 0.8},
        'original': {'name': 'Reds', 'start': 0.8, 'end': 1}
    }

    _PLOT_STYLE = {
        'synthetic': {'alpha': 0.6, 'lw': 3},
        'original': {'alpha': 1.0, 'lw': 3}
    }

    def __init__(self, original: List = None, synthetic: List = None, processed: bool = False):
        self.original = original
        self.synthetic = synthetic
        self._processed: bool = processed
        self.images = {}

    @classmethod
    def from_dict(cls, data_: Dict):
        return cls(
            original=data_['original'],
            synthetic=data_['synthetic']
        )

    def process(self):
        if not self._processed:
            if self.synthetic:
                synthetic_results = self._process_batch(self.synthetic)
                assert len(synthetic_results) == len(self.synthetic)
                self.synthetic = synthetic_results
            if self.original:
                original_results = self._process_batch(self.original)
                assert len(original_results) == len(self.original)
                self.original = original_results
            self._processed = True
        return self

    @staticmethod
    def _process_batch(graphs):
        processor = MultifractalProcessor(graphs)
        processor.analyze()
        return processor.get_summary_data()

    def plot(self):
        self.images.update({
            'spectra': self.plot_spectra(),
            'dimensions': self.plot_dimensions(),
        })

    def plot_spectra(self):
        return self._create_plot('al_list', 'fal_list',
                                 r'$\alpha$ (Hölder Exponent)',
                                 r'$f(\alpha)$ (Multifractal Spectrum)')

    def plot_dimensions(self):
        return self._create_plot('valid_q', 'dim_list',
                                 r'Distorting Exponent $q$',
                                 r'Generalized Fractal Dimension $D(q)$')

    def _create_plot(self, x_key, y_key, x_label, y_label):
        plt.rcParams.update(self._PLOT_CONFIG)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        legend = []

        for data_type in ['synthetic', 'original']:
            results = getattr(self, f'{data_type}', [])
            assert results, f"No {data_type} data available."

            self._plot_dataset(
                ax=ax,
                entries=results,
                data_type=data_type,
                x_key=x_key,
                y_key=y_key,
                legend=legend
            )

        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.legend(handles=legend, loc='upper right', frameon=False)

        return finalize_plot(fig)

    def _plot_dataset(self, ax, entries, data_type, x_key, y_key, legend):
        cmap_config = self._CMAP_RANGES[data_type]
        style = self._PLOT_STYLE[data_type]
        cmap = plt.get_cmap(cmap_config['name'])

        color_values = np.linspace(
            cmap_config['start'],
            cmap_config['end'],
            len(entries)
        )

        for idx, entry in enumerate(entries):
            ax.plot(
                entry[x_key],
                entry[y_key],
                color=cmap(color_values[idx]),
                alpha=style['alpha'],
                lw=style['lw']
            )

        legend.append(Line2D(
            [0], [0],
            color=cmap(np.mean(color_values)),
            lw=4,
            label=f"{data_type.capitalize()} (n={len(entries)})"
        ))
