import os
import pickle
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from analysis.multifractal_analyzer import MultifractalAnalyzer


class MultifractalAnalysisProcessor:
    def __init__(self,
                 graphs: List[nx.Graph],
                 names: Optional[List[str]] = None,
                 weighted: bool = False,
                 digit_round: int = 0):

        self.graphs = graphs
        self.names = names if names else [f"Graph_{i}" for i in range(len(graphs))]
        self.weighted = weighted
        self.digit_round = digit_round

        self.tau_lists = []
        self.centralities = []
        self.summary = pd.DataFrame()

    def analyze_all(self):
        holder_exp_list = []
        width_list = []
        dimension_diff_list = []
        avg_nfd_list = []
        avg_closeness_list = []
        avg_degree_list = []
        avg_clustering_list = []
        avg_betweenness_list = []
        avg_orc_list = []
        assortativity_list = []
        avg_eigen_list = []
        diameter_list = []

        for i, G in enumerate(self.graphs):
            name = self.names[i]
            analyzer = MultifractalAnalyzer(G, digit_round=self.digit_round)

            # Multi-fractal analysis
            tau_list = analyzer.compute_multifractal_taus()
            self.tau_lists.append(tau_list)

            # n-spectrum
            alpha_0, width, al_list, fal_list = analyzer.compute_n_spectrum(tau_list)
            holder_exp_list.append(alpha_0)
            width_list.append(width)

            # dimension D(q)
            dim_vals, dmax, dmin, ddiff = analyzer.compute_n_dimension(tau_list)
            dimension_diff_list.append(ddiff)

            # centralities
            cdict = analyzer.compute_centralities()
            avg_nfd_list.append(np.mean(cdict['nfd']))
            avg_closeness_list.append(np.mean(cdict['closeness']))
            avg_degree_list.append(np.mean(cdict['degree']))
            avg_clustering_list.append(np.mean(cdict['clustering']))

            # betweenness
            bt = analyzer.compute_betweenness()
            avg_betweenness_list.append(np.mean(bt))

            # orc
            curv = analyzer.compute_ollivier_ricci_curvature(alpha=0.5)
            avg_orc_list.append(np.mean(curv))

            # assortativity
            assort = analyzer.compute_assortativity()
            assortativity_list.append(assort)

            # eigen
            eigenvals = analyzer.compute_eigenvector_centrality()
            avg_eigen_list.append(np.mean(eigenvals) if len(eigenvals) > 0 else 0)

            # diameter
            diam = analyzer.compute_diameter()
            diameter_list.append(diam)

            print(f"Finished analysis of {name}.")

        self.summary = pd.DataFrame({
            'Graph': self.names,
            'HolderExp': holder_exp_list,
            'Width': width_list,
            'DimensionDiff': dimension_diff_list,
            'AvgFracDim': avg_nfd_list,
            'AvgCloseness': avg_closeness_list,
            'AvgDegree': avg_degree_list,
            'AvgClustering': avg_clustering_list,
            'AvgBetweenness': avg_betweenness_list,
            'AvgORCurv': avg_orc_list,
            'Assortativity': assortativity_list,
            'AvgEigen': avg_eigen_list,
            'Diameter': diameter_list
        })

    def save_summary(self, path: str):
        if self.summary.empty:
            print("Warning: Summary is empty. Did you run analyze_all()?")
        else:
            self.summary.to_csv(path, index=False)
            print(f"Summary CSV saved to {path}")

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
