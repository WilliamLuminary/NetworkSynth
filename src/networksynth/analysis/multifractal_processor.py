# SPDX-License-Identifier: GPL-3.0-or-later
import logging
from typing import Dict, List, Optional, Union

import numpy as np

from networksynth.graphs.synth_graph import SynthGraph

from .multifractal_analyzer import MultifractalAnalyzer

logger = logging.getLogger(__name__)


def _worker_count(requested: Optional[int], num_jobs: int) -> int:
    if num_jobs <= 1:
        return 1
    if requested is not None:
        return max(1, min(requested, num_jobs))
    from networksynth.utils import worker_count

    return worker_count(num_jobs)


def _analyze_one(job) -> Dict:
    idx, graph, measure_weighted, full_q_band = job

    result = MultifractalAnalyzer(graph, measure_weighted, full_q_band).analyze_graph()
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
        **MultifractalProcessor._calculate_averages(result),
    }


class MultifractalProcessor:
    def __init__(
        self,
        arg: Union[Dict, SynthGraph, List[Dict], List[SynthGraph]],
        measure_weighted: bool,
        full_q_band: bool,
        max_workers: Optional[int] = None,
    ):
        self._measure_weighted = measure_weighted
        self._full_q_band = full_q_band
        self._max_workers = max_workers
        self._graphs: Optional[List[SynthGraph]] = None
        self._analysis_results: Optional[List[Dict]] = None
        if isinstance(arg, SynthGraph):
            self._graphs = [arg]
        elif isinstance(arg, Dict):
            self._analysis_results: List[Dict] = [arg]
        elif isinstance(arg[0], SynthGraph):
            self._graphs = arg
        elif isinstance(arg[0], Dict):
            self._analysis_results: List[Dict] = arg

    def get_summary_data(self) -> List[Dict]:
        return self._analysis_results

    def analyze(self) -> None:
        if self._analysis_results:
            logger.info("Augmenting with averages.")
            self._augment_with_averages()
        elif self._graphs:
            logger.info("Performing full analysis.")
            self._analysis_results = self._perform_full_analysis()

    def _perform_full_analysis(self) -> List[Dict]:
        jobs = [
            (i, g, self._measure_weighted, self._full_q_band)
            for i, g in enumerate(self._graphs)
        ]
        workers = _worker_count(self._max_workers, len(jobs))
        if workers == 1:
            return [_analyze_one(job) for job in jobs]

        from concurrent.futures import ProcessPoolExecutor

        from networksynth.utils import spawn_context

        logger.info(f"Analysing {len(jobs)} networks across {workers} processes.")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=spawn_context()
        ) as pool:
            return list(pool.map(_analyze_one, jobs))

    def _augment_with_averages(self) -> None:
        if self._has_averages(self._analysis_results[0]):
            return
        for entry in self._analysis_results:
            entry.update(self._calculate_averages(entry))

    @staticmethod
    def _has_averages(entry: Dict) -> bool:
        avg_keys = {
            "avg_nfd",
            "avg_closeness",
            "avg_degree",
            "avg_clustering",
            "avg_betweenness",
            "avg_ricci",
            "avg_eigen",
        }
        return avg_keys.issubset(entry.keys())

    @staticmethod
    def _calculate_averages(data_: Dict) -> Dict:
        def safe_mean(values):
            return float(np.nanmean(values)) if values else float("nan")

        return {
            "avg_nfd": safe_mean(data_.get("nfd_dist")),
            "avg_closeness": safe_mean(data_.get("closeness_dist")),
            "avg_degree": safe_mean(data_.get("degree_dist")),
            "avg_clustering": safe_mean(data_.get("clustering_dist")),
            "avg_betweenness": safe_mean(data_.get("betweenness_dist")),
            "avg_ricci": safe_mean(data_.get("ricci_dist")),
            "avg_eigen": safe_mean(data_.get("eigen_dist")),
        }
