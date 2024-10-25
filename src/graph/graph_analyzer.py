# graph_analyzer.py

import math
import numpy as np
import networkx as nx
import networkit as nk
from collections import Counter
from scipy.stats import linregress


class GraphAnalyzer:
    def __init__(self, graph):
        self.graph = graph

    def calculate_multifractal_spectrum(self, Q, weight=True, fdigi=0):
        N_list = []
        r_g_all_set = set()
        graph = nx.convert_node_labels_to_integers(self.graph)
        if weight:
            G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight')
        else:
            G_nk = nk.nxadapter.nx2nk(graph)

        for node in graph.nodes():
            distances = (nk.distance.Dijkstra(G_nk, node, storePaths=False)).run().getDistances()
            grow = [d for d in distances if 0 < d < 99999]
            if fdigi == 0:
                grow = [math.ceil(d) for d in grow]
            else:
                grow = [round(d, fdigi) for d in grow if round(d, fdigi) != 0]
            num = Counter(grow)
            r_g_all_set.update(num.keys())
            N_list.append(num)

        r_g_all = np.array(sorted(list(r_g_all_set)))
        diameter = r_g_all[-1]
        Zq_list = []

        for q in Q:
            Zq = np.zeros(len(r_g_all))
            for idx, num in enumerate(N_list):
                N_vals = np.array([sum([count for radius, count in num.items() if radius <= r]) for r in r_g_all])
                Zq += (N_vals / N_vals[-1]) ** q
            Zq_list.append(Zq)

        tau_list = []
        for idx, q in enumerate(Q):
            x = np.log(r_g_all / diameter)
            y = np.log(Zq_list[idx])
            slope, _, _, _, _ = linregress(x, y)
            tau_list.append(slope)

        return tau_list

    def n_spectrum(self, tau_list, Q):
        al_list = []
        fal_list = []
        for i in range(1, len(Q)):
            al = (tau_list[i] - tau_list[i - 1]) / (Q[i] - Q[i - 1])
            al_list.append(al)
        for j in range(len(Q) - 1):
            fal = Q[j] * al_list[j] - tau_list[j]
            fal_list.append(fal)
        alpha_0 = al_list[np.argmax(fal_list)]
        width = np.max(al_list) - np.min(al_list)
        return alpha_0, width, al_list, fal_list
