# src/handlers/single_graph_multifractal_analyzer.py
from typing import Tuple

import math
import numpy as np
import networkx as nx
import networkit as nk
from collections import Counter

from matplotlib import pyplot as plt
from scipy.stats import linregress

from config import Config
from utils import figure_to_ndarray

PLOT_MULTIFRACTAL_SPECTRUM = True


class SingleGraphMultifractalAnalyzer:
    Q = [q / 100 for q in range(-300, 301, 10)]

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weighted = Config.MEASURE_WEIGHTED
        self.f_digit = 0

    def multifractal_analysis(self) -> Tuple[float, float]:
        tau_list, r_g_all, diameter, zq_list = self.calculate_multifractal_taus()
        alpha_0, width, al_list, fal_list = self.n_spectrum(tau_list)
        return alpha_0, width

    def calculate_multifractal_taus(self):
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

    def plot_multifractal_spectrum(self, r_g_all, diameter, zq_list):
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

    @staticmethod
    def n_spectrum(tau_list):
        al_list = []
        fal_list = []
        for i in range(1, len(SingleGraphMultifractalAnalyzer.Q)):
            al = (tau_list[i] - tau_list[i - 1]) / (SingleGraphMultifractalAnalyzer.Q[i] - SingleGraphMultifractalAnalyzer.Q[i - 1])
            al_list.append(al)
        for j in range(len(SingleGraphMultifractalAnalyzer.Q) - 1):
            fal = SingleGraphMultifractalAnalyzer.Q[j] * al_list[j] - tau_list[j]
            fal_list.append(fal)
        alpha_0 = al_list[np.argmax(fal_list)]
        width = np.max(al_list) - np.min(al_list)
        return alpha_0, width, al_list, fal_list
