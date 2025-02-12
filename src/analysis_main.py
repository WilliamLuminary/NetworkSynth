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
        'z_order': 1
    },
    'original': {
        'cmap': 'Reds',
        'alpha': 0.7,
        'lw': 2.5,
        'z_order': 2
    }
}


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


def load_data(result_dir) -> Dict[str, Dict[str, List[nx.Graph]]]:
    base_output_dir = Config.BASE_OUTPUT_PATH
    base_directory = os.path.abspath(os.path.join(base_output_dir, result_dir))
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
        entries = summary_data[data_type]
        cmap = plt.get_cmap(style['cmap'])

        color_vals = np.linspace(0.2, 0.8, len(entries))

        for idx, entry in enumerate(entries):
            color = cmap(color_vals[idx])
            ax.plot(entry["al_list"], entry["fal_list"],
                    color=color,
                    alpha=style['alpha'],
                    lw=style['lw'],
                    zorder=style['z_order'])

        legend_handles.append(
            plt.Line2D([0], [0], color=cmap(0.5),
                       lw=3, label=f"{data_type.capitalize()} (n={len(entries)})")
        )

    ax.set_xlabel(r'$\alpha$ (Hölder Exponent)', fontweight='bold')
    ax.set_ylabel(r'$f(\alpha)$ (Multifractal Spectrum)', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=False, fontsize=18)

    return finalize_plot(fig)


def plot_n_dimensions(summary_data: Dict[str, List[Dict]]) -> np.ndarray:
    fig, ax = setup_plot()
    legend_handles = []

    for data_type in summary_data:
        style = DATA_STYLE[data_type]
        entries = summary_data[data_type]
        cmap = plt.get_cmap(style['cmap'])

        color_vals = np.linspace(0.2, 0.8, len(entries))

        for idx, entry in enumerate(entries):
            color = cmap(color_vals[idx])
            ax.plot(entry["valid_q"], entry["dim_list"],
                    color=color,
                    alpha=style['alpha'],
                    lw=style['lw'],
                    zorder=style['z_order'])

        legend_handles.append(
            plt.Line2D([0], [0], color=cmap(0.5),
                       lw=3, label=f"{data_type.capitalize()} (n={len(entries)})")
        )

    ax.set_xlabel(r'Distorting Exponent $q$', fontweight='bold')
    ax.set_ylabel(r'Generalized Fractal Dimension $D(q)$', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=False, fontsize=18)

    return finalize_plot(fig)


def finalize_plot(fig):
    plt.tight_layout()
    plt.show()
    fig = figure_to_ndarray(fig)
    plt.close()
    return fig


Config.MEASURE_WEIGHTED = False
if __name__ == '__main__':
    data = load_data('results_20250202_013241')

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
        image1 = plot_spectra_from_summary(data_dict)
        image2 = plot_n_dimensions(data_dict)
