import os
import pickle
from collections import deque
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

    def __init__(self, original: List = None, synthetic: List=None, processed: bool = False):
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
            results = getattr(self, f'{data_type}_results', [])
            if not results:
                return None

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


def _load_graphs_pkl(folder: str) -> List[nx.Graph]:
    for file in os.listdir(folder):
        if file.endswith('.pkl') and 'network' in file:
            with open(os.path.join(folder, file), 'rb') as f:
                content = pickle.load(f)
                return [content] if isinstance(content, nx.Graph) else content
    return []


def find_directories(base_dir: str, sub_folders=('synthetic', 'origin', 'original'), max_depth: int = 3) -> Dict[
    str, Dict]:
    name_networks_dict = {}
    queue = deque([(base_dir, 0, '')])  # (path, depth, rel_path)

    while queue:
        current_dir, depth, rel_path = queue.popleft()

        if os.path.basename(current_dir).startswith(('.', '__')):
            continue

        found = []
        tmp_dict = {}
        for entry in os.listdir(current_dir):
            if entry in sub_folders:
                key = 'original' if entry == 'origin' else entry
                full_path = os.path.join(current_dir, entry)
                tmp_dict[key] = _load_graphs_pkl(full_path)
                found.append(entry)
        if tmp_dict:
            name_networks_dict[rel_path] = tmp_dict

        if not found and depth < max_depth:
            for entry in os.listdir(current_dir):
                entry_path = os.path.join(current_dir, entry)
                if os.path.isdir(entry_path):
                    new_rel = os.path.join(rel_path, entry) if rel_path else entry
                    queue.append((entry_path, depth + 1, new_rel))
        elif found:
            name_networks_dict[rel_path]['path'] = current_dir

    return name_networks_dict


def load_data(result_dir: str) -> Dict[str, Dict]:
    base_output_dir = Config.BASE_OUTPUT_PATH
    base_directory = os.path.abspath(os.path.join(base_output_dir, result_dir))
    return find_directories(base_directory, ('synthetic', 'origin', 'original'))


ConfigSample.initialize()
if __name__ == '__main__':
    data = load_data('results_20250202_013241')

    for set_name, dataset in data.items():
        batch_processor = MultifractalBatchProcessor(dataset).process()
        batch_processor.plot_spectra()
        batch_processor.plot_dimensions()
