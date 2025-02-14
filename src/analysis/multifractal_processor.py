import logging
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np

from analysis.multifractal_analyzer import MultifractalAnalyzer

logger = logging.getLogger(__name__)


class MultifractalProcessor:
    def __init__(self, arg: Union[Dict, nx.Graph, List[Dict], List[nx.Graph]]):
        self._graphs: Optional[List[nx.Graph]] = None
        self._analysis_results: Optional[List[Dict]] = None
        if isinstance(arg, nx.Graph):
            self._graphs = [arg]
        elif isinstance(arg, Dict):
            self._analysis_results: List[Dict] = [arg]
        elif isinstance(arg[0], nx.Graph):
            self._graphs = arg
        elif isinstance(arg[0], Dict):
            self._analysis_results: List[Dict] = arg

    def get_summary_data(self) -> List[Dict]:
        return self._analysis_results

    def analyze(self) -> None:
        if self._analysis_results:
            self._augment_with_averages()
        elif self._graphs:
            self._analysis_results = self._perform_full_analysis()

    def _perform_full_analysis(self) -> List[Dict]:
        return [self._analyze_single_graph(i, G)
                for i, G in enumerate(self._graphs)]

    def _analyze_single_graph(self, idx: int, graph: nx.Graph) -> Dict:
        analyzer = MultifractalAnalyzer(graph)
        result = analyzer.analyze_graph()

        return {
            "graph_idx": idx,
            "tau_list": result["tau_list"],
            "al_list": result["al_list"],
            "fal_list": result["fal_list"],
            "dim_list": result["dim_list"],
            "dim_diff": result["dim_diff"],
            "valid_q": result["valid_q"],
            "diameter": result["diameter"],
            "holder_exp": result["alpha_0"],
            "width": result["width"],
            "assortativity": result["assortativity"],
            **self._calculate_averages(result)
        }

    def _augment_with_averages(self) -> None:
        if self._has_averages(self._analysis_results[0]):
            return
        for entry in self._analysis_results:
            entry.update(self._calculate_averages(entry))

    @staticmethod
    def _has_averages(entry: Dict) -> bool:
        avg_keys = {'avg_nfd', 'avg_closeness', 'avg_degree',
                    'avg_clustering', 'avg_betweenness',
                    'avg_ricci', 'avg_eigen'}
        return avg_keys.issubset(entry.keys())

    @staticmethod
    def _calculate_averages(data_: Dict) -> Dict:
        def safe_mean(values):
            return float(np.nanmean(values)) if values else float('nan')

        return {
            "avg_nfd": safe_mean(data_.get("nfd_dist")),
            "avg_closeness": safe_mean(data_.get("closeness_dist")),
            "avg_degree": safe_mean(data_.get("degree_dist")),
            "avg_clustering": safe_mean(data_.get("clustering_dist")),
            "avg_betweenness": safe_mean(data_.get("betweenness_dist")),
            "avg_ricci": safe_mean(data_.get("ricci_dist")),
            "avg_eigen": safe_mean(data_.get("eigen_dist")),
        }
