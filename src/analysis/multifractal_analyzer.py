# src/handlers/multifractal_analyzer.py
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import math
import networkit as nk
import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import linregress

from config import Config
from utils import figure_to_ndarray
from utils.utils import keep_largest_connected_component


@dataclass
class MultifractalErrorFeatures:
    holder_exponent: float
    spectrum_width: float


class MultifractalAnalyzer:
    Q = [q / 100 for q in range(-300, 301, 10)]

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weighted = Config.MEASURE_WEIGHTED
        self.f_digit = 0

    def analyze_error_values(self) -> MultifractalErrorFeatures:
        tau_list, _, _, _ = self.compute_multifractal_taus()
        alpha_0, width, _, _ = self.compute_n_spectrum(tau_list)
        error_features = MultifractalErrorFeatures(alpha_0, width)
        return error_features

    @staticmethod
    def analyze_error(this: MultifractalErrorFeatures, other: MultifractalErrorFeatures) -> float:
        return euclidean([this.holder_exponent, this.spectrum_width], [other.holder_exponent, other.spectrum_width])

    def compute_multifractal_taus(self):
        graph = nx.convert_node_labels_to_integers(self.graph)
        if self.weighted:
            G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight')
        else:
            G_nk = nk.nxadapter.nx2nk(graph)

        N_list = []
        r_g_all_set = set()
        for node in graph.nodes():
            # noinspection PyUnresolvedReferences
            distances = nk.distance.Dijkstra(G_nk, node, storePaths=False).run().getDistances()
            grow = [d for d in distances if 0 < d < 99999]
            grow.sort()
            if self.f_digit == 0:
                grow = [math.ceil(d) for d in grow]
            else:
                grow = [round(d, self.f_digit) for d in grow if round(d, self.f_digit) != 0]
            num = Counter(grow)
            r_g_all_set.update(num.keys())
            N_list.append(num)

        r_g_all = np.array(sorted(list(r_g_all_set)))
        if len(r_g_all) < 2:
            return [0.0 for _ in self.Q]

        diameter = r_g_all[-1]
        Nw_mat = np.ones((len(N_list), len(r_g_all)))
        for i, num_counter in enumerate(N_list):
            for j, r_val in enumerate(r_g_all):
                Nw_mat[i, j] += sum(count for dist_, count in num_counter.items() if dist_ <= r_val)

        zq_list = []
        for q in self.Q:
            row_sums = []
            for i in range(len(Nw_mat)):
                if Nw_mat[i, -1] == 0:
                    row_sums.append(0)
                else:
                    row_sums.append((Nw_mat[i, :] / Nw_mat[i, -1]) ** q)
            zq = np.sum(row_sums, axis=0)
            zq_list.append(zq)

        tau_list = []
        for idx, q in enumerate(self.Q):
            x = np.log(r_g_all / diameter)
            y = np.log(zq_list[idx])
            slope, _, _, _, _ = linregress(x, y)
            tau_list.append(slope)
        return tau_list, r_g_all, diameter, zq_list

    def plot_multifractal_taus(self, r_g_all, diameter, zq_list) -> np.ndarray:
        fig, ax = plt.figure(figsize=(8, 8), dpi=300)
        for idx, q in enumerate(self.Q):
            x = np.log(r_g_all / diameter)
            y = np.log(zq_list[idx])
            plt.plot(x, y, '*', label=f'q={q:.0f}')
            plt.xlabel('ln(r/d)')
            plt.ylabel('ln(sum function)')
        plt.legend()
        fig = figure_to_ndarray(fig)
        plt.close()
        return fig

    def compute_n_spectrum(self, tau_list):
        al_list = []
        fal_list = []
        for i in range(1, len(self.Q)):
            al = (tau_list[i] - tau_list[i - 1]) / (self.Q[i] - self.Q[i - 1])
            al_list.append(al)
        for j in range(len(self.Q) - 1):
            fal = self.Q[j] * al_list[j] - tau_list[j]
            fal_list.append(fal)
        alpha_0 = al_list[np.argmax(fal_list)]
        width = np.max(al_list) - np.min(al_list)
        return alpha_0, width, al_list, fal_list

    @staticmethod
    def plot_n_spectrum(al_list, fal_list, label, color) -> np.ndarray:
        fig, ax = plt.figure(figsize=(8, 8), dpi=300)
        plt.plot(al_list, fal_list, label=label, linewidth=3, color=color)
        plt.xlabel('Lipschitz-Hölder exponent, 'r'$\alpha$')
        plt.ylabel('Multi-fractal spectrum, 'r'$f(\alpha)$')
        plt.legend()
        fig = figure_to_ndarray(fig)
        plt.close()
        return fig

    def compute_n_dimension(self, tau_list):
        q_list = self.Q
        valid_pairs = [(q, tau) for q, tau in zip(q_list, tau_list) if q != 0]
        valid_q, valid_tau = zip(*valid_pairs)
        dim_list = [tau / q for q, tau in zip(valid_q, valid_tau)]

        dim_max = np.max(dim_list)
        dim_min = np.min(dim_list)
        diff = dim_max - dim_min
        return dim_list, dim_max, dim_min, diff, list(valid_q)

    @staticmethod
    def plot_n_dimension(dim_list, valid_q, label, color) -> np.ndarray:
        fig, ax = plt.figure(figsize=(8, 8), dpi=300)
        plt.plot(valid_q, dim_list, label=label, linewidth=3, color=color)
        plt.xlabel('Distorting exponent, 'r'$q$')
        plt.ylabel('Generalized fractal dimension, 'r'$D(q)$')
        fig = figure_to_ndarray(fig)
        plt.close()
        return fig

    def _compute_node_dimension(self) -> Dict[int, float]:
        graph = nx.convert_node_labels_to_integers(self.graph)
        if self.weighted:
            for _, _, d in graph.edges(data=True):
                if d.get('weight', 0) != 0:
                    d['weight'] = 1.0 / d['weight']
            G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight')
        else:
            G_nk = nk.nxadapter.nx2nk(graph)

        node_dimensions = {}
        for node in graph.nodes():
            # noinspection PyUnresolvedReferences
            distances = nk.distance.Dijkstra(G_nk, int(node), storePaths=False).run().getDistances()
            distances.sort()
            if self.weighted:
                distances = [round(d, self.f_digit) for d in distances if round(d, self.f_digit) != 0]

            dist_count = Counter(distances)
            unique_dist, cum_count = [], []
            s = 0
            for dist_val, count in dist_count.items():
                s += count
                if dist_val > 0:
                    unique_dist.append(dist_val)
                    cum_count.append(s)

            if len(unique_dist) > 2:
                x = np.log(unique_dist)
                y = np.log(cum_count)
                slope, _, _, _, _ = linregress(x, y)
                node_dimensions[node] = slope
            else:
                node_dimensions[node] = 0
        return node_dimensions

    def compute_centralities(self) -> Dict[str, List[float]]:
        if self.weighted:
            closeness_distance = lambda u, v, d: 1 / d['weight']
            degree_attr = 'weight'
            clustering_attr = 'weight'
        else:
            closeness_distance = None
            degree_attr = None
            clustering_attr = None

        nfd_centrality = self._compute_node_dimension()
        closeness_centrality = nx.closeness_centrality(self.graph, distance=closeness_distance)
        degree_centrality = dict(self.graph.degree(weight=degree_attr))
        cluster_coef = nx.clustering(self.graph, weight=clustering_attr)

        return {
            'nfd': list(nfd_centrality.values()),
            'closeness': list(closeness_centrality.values()),
            'degree': list(degree_centrality.values()),
            'clustering': list(cluster_coef.values())
        }

    def compute_betweenness(self) -> List[float]:
        if self.weighted:
            graph_copy = self.graph.copy()
            for _, _, d in graph_copy.edges(data=True):
                d['weight'] = 1.0 / d['weight'] if d.get('weight', 0) != 0 else float('inf')
            G_nk = nk.nxadapter.nx2nk(graph_copy, weightAttr='weight')
        else:
            G_nk = nk.nxadapter.nx2nk(self.graph, weightAttr=None)
        # noinspection PyUnresolvedReferences
        bt = nk.centrality.Betweenness(G_nk, normalized=True).run().scores()
        return bt

    def compute_ollivier_ricci_curvature(self) -> List[float]:
        graph_copy = self.graph.copy()
        if self.weighted:
            for u, v, d in graph_copy.edges(data=True):
                if d.get('weight', 0) != 0:
                    d['weight'] = 1.0 / d['weight']
            orc = OllivierRicci(nx.convert_node_labels_to_integers(graph_copy),
                                alpha=.5, verbose="ERROR", weight='weight')
        else:
            orc = OllivierRicci(nx.convert_node_labels_to_integers(graph_copy),
                                alpha=.5, verbose="ERROR", weight=None)
        orc.compute_ricci_curvature()
        return [d['ricciCurvature'] for _, _, d in orc.G.edges(data=True)]

    def compute_assortativity(self) -> float:
        if self.weighted:
            return nx.degree_pearson_correlation_coefficient(self.graph, weight='weight')
        else:
            return nx.degree_pearson_correlation_coefficient(self.graph)

    def compute_eigenvector_centrality(self) -> List[float]:
        if nx.is_connected(self.graph):
            G_lcc = self.graph
        else:
            G_lcc = keep_largest_connected_component(self.graph)

        try:
            if self.weighted:
                ec_dict = nx.eigenvector_centrality(G_lcc, max_iter=1000, weight='weight')
            else:
                ec_dict = nx.eigenvector_centrality(G_lcc, max_iter=1000)
            return list(ec_dict.values())
        except nx.PowerIterationFailedConvergence:
            return [float('nan')] * G_lcc.number_of_nodes()

    def compute_diameter(self) -> float:
        if nx.is_connected(self.graph):
            return nx.diameter(self.graph)
        else:
            return nx.diameter(keep_largest_connected_component(self.graph))

    def analyze_graph(self) -> Dict[str, List]:
        self.f_digit = 1
        tau_list, r_g_all, diameter, zq_list = self.compute_multifractal_taus()
        self.f_digit = 0

        alpha_0, width, al_list, fal_list = self.compute_n_spectrum(tau_list)

        self.f_digit = 2
        dim_list, dim_max, dim_min, dim_diff, valid_q = self.compute_n_dimension(tau_list)
        self.f_digit = 0

        centralities = self.compute_centralities()
        betweenness = self.compute_betweenness()
        ricci_list = self.compute_ollivier_ricci_curvature()
        assort = self.compute_assortativity()
        eigen_list = self.compute_eigenvector_centrality()
        diam = self.compute_diameter()

        return {
            # Graph-level multifractal
            "tau_list": tau_list,
            "alpha_0": alpha_0,
            "width": width,
            "dim_diff": dim_diff,
            "diameter": diam,
            "assortativity": assort,

            # Node-level distributions
            "nfd_dist": centralities["nfd"],
            "closeness_dist": centralities["closeness"],
            "degree_dist": centralities["degree"],
            "clustering_dist": centralities["clustering"],
            "betweenness_dist": betweenness,
            "ricci_dist": ricci_list,
            "eigen_dist": eigen_list
        }
