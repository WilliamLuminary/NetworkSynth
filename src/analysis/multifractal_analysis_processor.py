import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from analysis.multifractal_analyzer import MultifractalAnalyzer
from config import Config
from utils import figure_to_ndarray

logger = logging.getLogger(__name__)


class MultifractalAnalysisProcessor:
    def __init__(self,
                 graphs: Union[List[nx.Graph], Dict[str, List[nx.Graph]]],
                 name: Optional[str] = None):
        if isinstance(graphs, dict):
            self.data_dict = graphs
        else:
            if not name:
                logger.warning("No name provided for the dataset. Using 'default' as the name.")
                name = "default"
            self.data_dict = {name: graphs}

        if self.data_dict:
            min_length = min(len(graph_list) for graph_list in self.data_dict.values())
            self.data_dict = {
                key: graph_list[:min_length] for key, graph_list in self.data_dict.items()
            }

        self.summary_data: Dict[str, List[Dict]] = {}

    def run_analysis(self) -> None:
        for name, graph_list in self.data_dict.items():
            graph_metrics_list = []

            for i, G in enumerate(graph_list):
                analyzer = MultifractalAnalyzer(G)
                result_dict = analyzer.analyze_graph()

                nfd_vals = result_dict["nfd_dist"]
                closeness_vals = result_dict["closeness_dist"]
                degree_vals = result_dict["degree_dist"]
                cluster_vals = result_dict["clustering_dist"]
                betweenness_vals = result_dict["betweenness_dist"]
                ricci_vals = result_dict["ricci_dist"]
                eigen_vals = result_dict["eigen_dist"]

                def safe_avg(arr):
                    return float(np.mean(arr)) if arr else float('nan')

                per_graph_data = {
                    "graph_idx": i,

                    "tau_list": result_dict["tau_list"],
                    "holder_exp": result_dict["alpha_0"],
                    "width": result_dict["width"],
                    # "alpha_0": result_dict["alpha_0"],
                    "al_list": result_dict["al_list"],
                    "fal_list": result_dict["fal_list"],

                    "dim_diff": result_dict["dim_diff"],
                    "diameter": result_dict["diameter"],
                    "assortativity": result_dict["assortativity"],

                    "AvgNFD": safe_avg(nfd_vals),
                    "AvgCloseness": safe_avg(closeness_vals),
                    "AvgDegree": safe_avg(degree_vals),
                    "AvgClustering": safe_avg(cluster_vals),
                    "AvgBetweenness": safe_avg(betweenness_vals),
                    "AvgRicci": safe_avg(ricci_vals),
                    "AvgEigen": safe_avg(eigen_vals),
                }
                graph_metrics_list.append(per_graph_data)

            self.summary_data[name] = graph_metrics_list


def get_script_dir():
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        return os.getcwd()


def load_pkl_files(base_dir, allowed_dirs) -> Dict[str, Dict[str, List[nx.Graph]]]:
    data_ = {str(d): {} for d in allowed_dirs}
    for root, dirs, files in os.walk(base_dir):
        current_dir = os.path.basename(root)
        if current_dir not in allowed_dirs:
            continue
        set_name = os.path.basename(os.path.dirname(root))
        for file in files:
            if file.endswith(".pkl") and "property" not in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    pkl_content = pickle.load(f)
                    if isinstance(pkl_content, nx.Graph):
                        pkl_content = [pkl_content]
                if set_name in data_[current_dir]:
                    print(f"Duplicate entry for {set_name} in {current_dir}, skipping.")
                else:
                    data_[current_dir][set_name] = pkl_content
    return data_


def load_data() -> Dict[str, Dict[str, List[nx.Graph]]]:
    script_dir = get_script_dir()
    result_name = 'results_20250202_013241'
    base_directory = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'output', result_name))
    return load_pkl_files(base_directory, allowed_dirs=('synthetic', 'origin', 'original'))


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

    proc_synthetic = MultifractalAnalysisProcessor(data['synthetic'])
    proc_synthetic.run_analysis()
    summary_synthetic = proc_synthetic.summary_data

    proc_origin = MultifractalAnalysisProcessor(data['origin'])
    proc_origin.run_analysis()
    summary_origin = proc_origin.summary_data

    summary = {}
    for ds_name, syn_list in summary_synthetic.items():
        if ds_name not in summary:
            summary[ds_name] = {}
        summary[ds_name]["synthetic"] = syn_list
    for ds_name, ori_list in summary_origin.items():
        if ds_name not in summary:
            summary[ds_name] = {}
        summary[ds_name]["original"] = ori_list

    for ds_name, data_dict in summary.items():
        image = plot_spectra_from_summary(data_dict)
        cv2.imwrite("1.png", image)
