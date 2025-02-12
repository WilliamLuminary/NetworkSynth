import os
import pickle
from typing import Dict, List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from analysis import MultifractalProcessor
from config import Config
from utils import figure_to_ndarray

# noinspection SpellCheckingInspection
PLOT_RC_PARAMS = {
    'font.size': 24,
    'axes.linewidth': 2,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.major.size': 8,
    'ytick.minor.size': 4
}
# noinspection SpellCheckingInspection
DATA_STYLE = {
    'synthetic': {
        'cmap': 'Blues',
        'alpha': 0.3,
        'lw': 1.5,
        'zorder': 1
    },
    'original': {
        'cmap': 'Reds',
        'alpha': 0.7,
        'lw': 2.5,
        'z_order': 2
    }
}


def get_script_dir():
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        return os.getcwd()


def load_pkl_files(base_dir, sub_folders) -> Dict[str, Dict[str, List[nx.Graph]]]:
    data_ = {}
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for sample_dir in os.listdir(base_dir):
        if sample_dir.startswith('__') or sample_dir.startswith('.'):
            continue
        sample_path = os.path.join(base_dir, sample_dir)
        if not os.path.isdir(sample_path):
            continue

        data_[sample_dir] = {}
        for sub in sub_folders:
            sub_path = os.path.join(sample_path, sub)
            if os.path.isdir(sub_path):
                graphs = []
                for file in os.listdir(sub_path):
                    if file.endswith(".pkl") and "property" not in file:
                        file_path = os.path.join(sub_path, file)
                        with open(file_path, 'rb') as f:
                            pkl_content = pickle.load(f)
                            if isinstance(pkl_content, nx.Graph):
                                pkl_content = [pkl_content]
                            graphs.extend(pkl_content)
                            break

                data_[sample_dir][sub] = graphs

    return data_


def load_data() -> Dict[str, Dict[str, List[nx.Graph]]]:
    script_dir = get_script_dir()
    result_dir_name = 'results_20250202_013241'
    base_directory = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'output', result_dir_name))
    return load_pkl_files(base_directory, sub_folders=('synthetic', 'origin', 'original'))


def setup_plot(fig_size=(10, 8), dpi=150):
    plt.rcParams.update(PLOT_RC_PARAMS)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    return fig, ax


def plot_spectra_from_summary(summary_data: Dict[str, List[Dict]]) -> np.ndarray:
    fig, ax = setup_plot()

    legend_handles = []

    for data_type in summary_data:
        style = DATA_STYLE[data_type]
        cmap = plt.get_cmap(style['cmap'])
        base_color = cmap(0.6)

        for entry in summary_data[data_type]:
            ax.plot(entry["al_list"], entry["fal_list"],
                    color=base_color,
                    alpha=style['alpha'],
                    lw=style['lw'],
                    zorder=style['z_order'])

        legend_handles.append(
            plt.Line2D([0], [0], color=base_color,
                       lw=3, label=data_type.capitalize())
        )

    ax.set_xlabel(r'$\alpha$ (Hölder Exponent)', fontweight='bold')
    ax.set_ylabel(r'$f(\alpha)$ (Multifractal Spectrum)', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=False, fontsize=18)

    return finalize_plot(fig, ax)


def plot_n_dimensions(summary_data: Dict[str, List[Dict]]) -> np.ndarray:
    """Plot generalized fractal dimensions with proper style handling"""
    fig, ax = setup_plot()

    legend_handles = []

    for data_type in summary_data:
        style = DATA_STYLE[data_type]
        cmap = plt.get_cmap(style['cmap'])
        base_color = cmap(0.6)  # Midpoint color

        for entry in summary_data[data_type]:
            ax.plot(entry["valid_q"], entry["dim_list"],
                    color=base_color,
                    alpha=style['alpha'],
                    lw=style['lw'],
                    zorder=style['z_order'])

        legend_handles.append(
            plt.Line2D([0], [0], color=base_color,
                       lw=3, label=data_type.capitalize())
        )

    ax.set_xlabel(r'Distorting Exponent $q$', fontweight='bold')
    ax.set_ylabel(r'Generalized Fractal Dimension $D(q)$', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=False, fontsize=18)

    return finalize_plot(fig, ax)


def finalize_plot(fig, _):
    plt.tight_layout()
    plt.show()
    fig = figure_to_ndarray(fig)
    plt.close()
    return fig


Config.MEASURE_WEIGHTED = False
if __name__ == '__main__':
    data = load_data()

    summary = {}
    for name_, data in data.items():
        current_summary = {}
        proc_synthetic = MultifractalProcessor(data['synthetic'])
        proc_synthetic.analyze()
        current_summary['synthetic'] = proc_synthetic.summary_data

        proc_origin = MultifractalProcessor(data['origin'])
        proc_origin.analyze()
        current_summary['original'] = proc_origin.summary_data

        summary[name_] = current_summary

    for ds_name, data_dict in summary.items():
        image = plot_spectra_from_summary(data_dict)
