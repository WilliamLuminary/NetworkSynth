import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from analysis.multifractal_analyzer import MultifractalAnalyzer

logger = logging.getLogger(__name__)


class MultifractalAnalysisProcessor:
    def __init__(self,
                 graphs: Union[List[nx.Graph], Dict[str, nx.Graph]],
                 name: Optional[str] = None):
        if isinstance(graphs, dict):
            self.data_dict = graphs
        else:
            if not name:
                logger.warning("No name provided for the dataset. Using 'default' as the name.")
                name = "default"
            self.data_dict = {name: graphs}

        self.summary_data: Dict[str, List[Dict]] = {}

    def run_analysis(self) -> None:
        for dataset_name, graphs in self.data_dict.items():
            self.summary_data[dataset_name] = []

            for i, G in enumerate(graphs):
                analyzer = MultifractalAnalyzer(G)
                result_dict = analyzer.analyze_graph()

                nfd_vals = result_dict["nfd_dist"]  # list of floats, one per node
                closeness_vals = result_dict["closeness_dist"]
                degree_vals = result_dict["degree_dist"]
                cluster_vals = result_dict["clustering_dist"]
                betweenness_vals = result_dict["betweenness_dist"]
                ricci_vals = result_dict["ricci_dist"]
                eigen_vals = result_dict["eigen_dist"]

                avg_nfd = float(np.mean(nfd_vals)) if len(nfd_vals) > 0 else float('nan')
                avg_closeness = float(np.mean(closeness_vals)) if len(closeness_vals) > 0 else float('nan')
                avg_degree = float(np.mean(degree_vals)) if len(degree_vals) > 0 else float('nan')
                avg_clustering = float(np.mean(cluster_vals)) if len(cluster_vals) > 0 else float('nan')
                avg_betweenness = float(np.mean(betweenness_vals)) if len(betweenness_vals) > 0 else float('nan')
                avg_ricci = float(np.mean(ricci_vals)) if len(ricci_vals) > 0 else float('nan')
                avg_eigen = float(np.mean(eigen_vals)) if len(eigen_vals) > 0 else float('nan')

                self.summary_data[dataset_name].append({
                    "Dataset": dataset_name,
                    "GraphIndex": i,
                    "HolderExp": result_dict["alpha_0"],
                    "Width": result_dict["width"],
                    "DimDiff": result_dict["dim_diff"],
                    "Diameter": result_dict["diameter"],
                    "Assortativity": result_dict["assortativity"],

                    "AvgNFD": avg_nfd,
                    "AvgCloseness": avg_closeness,
                    "AvgDegree": avg_degree,
                    "AvgClustering": avg_clustering,
                    "AvgBetweenness": avg_betweenness,
                    "AvgRicci": avg_ricci,
                    "AvgEigen": avg_eigen
                })

    @staticmethod
    def load_graphs_from_dir(base_dir: str) -> Tuple[List[nx.Graph], List[str]]:
        graphs = []
        names = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, list):
                        for idx, g in enumerate(data):
                            graphs.append(g)
                            names.append(f"{file}_idx{idx}")
                    elif isinstance(data, nx.Graph):
                        graphs.append(data)
                        names.append(file)
        return graphs, names
