import os
import pickle
from typing import Dict, List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from analysis import MultifractalProcessor
from config import Config, ConfigSample
from utils import figure_to_ndarray


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

    def __init__(self, data_: Dict):
        self.synthetic: List = data_['synthetic']
        self.original: List = data_['original']
        self._processed = False

    def process(self):
        if not self._processed:
            synthetic_results = self._process_batch(self.synthetic)
            assert len(synthetic_results) == len(self.synthetic)
            self.synthetic = synthetic_results

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
            results = getattr(self, f'{data_type}_results', [])
            if not results:
                continue

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

        return self._finalize_plot(fig)

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

    @staticmethod
    def _finalize_plot(fig):
        plt.tight_layout()
        img = figure_to_ndarray(fig)
        plt.close()
        return img


def load_pkl_files(base_dir, sub_folders) -> Dict[str, Dict[str, List[nx.Graph]]]:
    data_ = {}
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for set_name_dir in os.listdir(base_dir):
        if set_name_dir.startswith('__') or set_name_dir.startswith('.'):
            continue
        set_name_path = os.path.join(base_dir, set_name_dir)
        if not os.path.isdir(set_name_path):
            continue

        data_[set_name_dir] = {}
        for sub in sub_folders:
            sub_path = os.path.join(set_name_path, sub)
            if os.path.isdir(sub_path):
                graphs = _load_graphs(sub_path)
                if sub == 'origin':
                    sub = 'original'
                data_[set_name_dir][sub] = graphs
    return data_


def _load_graphs(folder):
    graphs = []
    for file in os.listdir(folder):
        if file.endswith('.pkl') and 'network' in file:
            with open(os.path.join(folder, file), 'rb') as f:
                content = pickle.load(f)
                if isinstance(content, nx.Graph):
                    graphs.append(content)
                else:
                    graphs.extend(content)
                break  # Currently, only one network file per directory
    return graphs


def load_data(result_dir: str) -> Dict[str, Dict[str, List[nx.Graph]]]:
    base_output_dir = Config.BASE_OUTPUT_PATH
    base_directory = os.path.abspath(os.path.join(base_output_dir, result_dir))
    return load_pkl_files(base_directory, sub_folders=('synthetic', 'origin', 'original'))


ConfigSample.initialize()
if __name__ == '__main__':
    data = load_data('results_20250202_013241')

    for set_name, dataset in data.items():
        batch_processor = MultifractalBatchProcessor(dataset).process()
        batch_processor.plot_spectra()
        batch_processor.plot_dimensions()
