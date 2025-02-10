import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx

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

                self.summary_data[dataset_name].append({
                    "GraphIndex": i,
                    "HolderExp": result_dict["holder_exp"],
                    "Width": result_dict["width"],
                    "DimDiff": result_dict["dim_diff"],
                    "AvgNFD": result_dict["avg_nfd"],
                    "AvgCloseness": result_dict["avg_closeness"],
                    "AvgDegree": result_dict["avg_degree"],
                    "AvgClustering": result_dict["avg_clustering"],
                    "AvgBetweenness": result_dict["avg_betweenness"],
                    "AvgRicci": result_dict["avg_ricci"],
                    "Assortativity": result_dict["assortativity"],
                    "AvgEigen": result_dict["avg_eigen"],
                    "Diameter": result_dict["diameter"],
                    "TauList": result_dict["tau_list"]
                })

                logger.info(
                    f"Processed dataset={dataset_name}, graph={i}, "
                    f"holder_exp={self.summary_data[dataset_name][-1]['HolderExp']:.4f}, "
                    f"width={self.summary_data[dataset_name][-1]['Width']:.4f}"
                )

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
