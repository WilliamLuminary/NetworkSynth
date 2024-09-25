import math
from collections import Counter

import networkx as nx
import networkit as nk
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import stats


def wnfd_nk(graph, Q, weight: bool = True, draw: bool = False, fdigi=0):
    N_list = []
    r_g_all_set = set()
    graph = nx.convert_node_labels_to_integers(graph)
    if weight:
        G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight')
    else:
        G_nk = nk.nxadapter.nx2nk(graph)

    for node in graph.nodes():
        grow = []
        grow_ori = (nk.distance.Dijkstra(G_nk, node, storePaths=False)).run().getDistances()

        for s in grow_ori:
            if 0 < s < 99999:
                grow.append(s)
        grow.sort()
        if fdigi == 0:
            grow = [math.ceil(d) for d in grow]
        else:
            grow = [round(d, fdigi) for d in grow if round(d, fdigi) != 0]
        num = Counter(grow)
        r_g_all_set.update(num.keys())
        N_list.append(num)

    r_g_all = np.array(sorted(list(r_g_all_set)))
    network_matrix = np.ones((len(N_list), len(r_g_all)))

    for i, num in enumerate(N_list):
        for j, r in enumerate(r_g_all):
            network_matrix[i, j] += sum(count for radius, count in num.items() if radius <= r)

    diameter = r_g_all[-1]
    Zq_list = []

    for q in Q:
        Zq_mat = np.power(network_matrix / network_matrix[:, -1, None], q)
        Zq_list.append(np.sum(Zq_mat, axis=0))

    tau_list = []
    if draw:
        plt.figure(figsize=(7, 7))
        for idx, q in enumerate(Q):
            r_g_all_np = np.array(r_g_all)
            x = np.log(r_g_all_np / diameter)
            y = np.log(Zq_list[idx])
            q = format(q, '.0f')
            plt.plot(x, y, '*', label='q=' + str(q))
            slope, intercept, _, _, _ = stats.linregress(x, y)
            tau_list.append(slope)
            plt.xlabel('ln(r/d)')
            plt.ylabel('ln(sum function)')
    else:
        for idx, q in enumerate(Q):
            r_g_all_np = np.array(r_g_all)
            x = np.log(r_g_all_np / diameter)
            y = np.log(Zq_list[idx])
            slope, intercept, _, _, _ = stats.linregress(x, y)
            tau_list.append(slope)

    return tau_list


def n_spectrum(tau_list, q_list):
    al_list = []
    fal_list = []
    for i in range(1, len(q_list)):
        al = (tau_list[i] - tau_list[i - 1]) / (q_list[i] - q_list[i - 1])
        al_list.append(al)
    for j in range(len(q_list) - 1):
        fal = q_list[j] * al_list[j] - tau_list[j]
        fal_list.append(fal)
    alpha_0 = al_list[np.argmax(fal_list)]
    width = np.max(al_list) - np.min(al_list)
    return alpha_0, width, al_list, fal_list
