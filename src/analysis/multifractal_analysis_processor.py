import logging
import os
import pickle
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from analysis.multifractal_analyzer import MultifractalAnalyzer
from config import Config, AnalysisMode
from utils import figure_to_ndarray

logger = logging.getLogger(__name__)


class MultifractalProcessor:
    def __init__(self, arg: Union[Dict, nx.Graph, List[Dict], List[nx.Graph]]):
        self.graphs: Optional[List[nx.Graph]] = None
        self.summary_data: Optional[List[Dict]] = None
        if isinstance(arg[0], nx.Graph):
            self.graphs = arg
        elif isinstance(arg[0], Dict):
            self.summary_data: List[Dict] = []
        elif isinstance(arg, nx.Graph):
            self.graphs = [arg]
        elif isinstance(arg, Dict):
            self.summary_data: List[Dict] = []

    def analyze(self) -> None:
        if self.graphs:
            self.summary_data = self._perform_full_analysis()
        elif self.summary_data:
            self._augment_with_averages()

    def _perform_full_analysis(self) -> List[Dict]:
        return [self._analyze_single_graph(i, G)
                for i, G in enumerate(self.graphs)]

    def _analyze_single_graph(self, idx: int, graph: nx.Graph) -> Dict:
        analyzer = MultifractalAnalyzer(graph)
        result = analyzer.analyze_graph()

        return {
            "graph_idx": idx,
            "tau_list": result["tau_list"],
            "al_list": result["al_list"],
            "fal_list": result["fal_list"],
            "dim_diff": result["dim_diff"],
            "diameter": result["diameter"],
            "holder_exp": result["alpha_0"],
            "width": result["width"],
            "assortativity": result["assortativity"],
            **self._calculate_averages(result)
        }

    def _augment_with_averages(self) -> None:
        if self._has_averages(self.summary_data[0]):
            return
        for entry in self.summary_data:
            entry.update(self._calculate_averages(entry))

    @staticmethod
    def _has_averages(entry: Dict) -> bool:
        avg_keys = {'avg_nfd', 'avg_closeness', 'avg_degree',
                    'avg_clustering', 'avg_betweenness',
                    'avg_ricci', 'avg_eigen'}
        return avg_keys.issubset(entry.keys())

    @staticmethod
    def _calculate_averages(data: Dict) -> Dict:
        def safe_mean(values):
            return float(np.nanmean(values)) if values else float('nan')

        return {
            "avg_nfd": safe_mean(data.get("nfd_dist")),
            "avg_closeness": safe_mean(data.get("closeness_dist")),
            "avg_degree": safe_mean(data.get("degree_dist")),
            "avg_clustering": safe_mean(data.get("clustering_dist")),
            "avg_betweenness": safe_mean(data.get("betweenness_dist")),
            "avg_ricci": safe_mean(data.get("ricci_dist")),
            "avg_eigen": safe_mean(data.get("eigen_dist")),
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


def plot_spectra_from_summary(data_dict,
                              synthetic_key='synthetic',
                              origin_key='original'):
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    synthetic_results = data_dict[synthetic_key]
    for idx, entry in enumerate(synthetic_results):
        al_list = entry["al_list"]
        fal_list = entry["fal_list"]
        holder_exp = entry["holder_exp"]
        width = entry["width"]
        graph_idx = entry["graph_idx"]
        plt.plot(al_list, fal_list, label=f"Synth#{graph_idx}",
                 linewidth=2, color='blue')

    origin_results = data_dict[origin_key]
    for idx, entry in enumerate(origin_results):
        al_list = entry["al_list"]
        fal_list = entry["fal_list"]
        holder_exp = entry["holder_exp"]
        width = entry["width"]
        graph_idx = entry["graph_idx"]
        plt.plot(al_list, fal_list, label=f"Origin#{graph_idx}",
                 linewidth=2, color='red')

    plt.xlabel(r'$\alpha$ (Hölder exponent)')
    plt.ylabel(r'$f(\alpha)$ (Multifractal spectrum)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig = figure_to_ndarray(fig, True)
    return fig


Config.MEASURE_WEIGHTED = False
if __name__ == '__main__':
    data = load_data()

    summary = {}
    for name_, data in data.items():
        current_summary = {}
        proc_synthetic = MultifractalProcessor(data['synthetic'])
        proc_synthetic.run_analysis()
        current_summary['synthetic'] = proc_synthetic.summary_data

        proc_origin = MultifractalProcessor(data['origin'])
        proc_origin.run_analysis()
        current_summary['original'] = proc_origin.summary_data

        summary[name_] = current_summary

    for ds_name, data_dict in summary.items():
        image = plot_spectra_from_summary(data_dict)
