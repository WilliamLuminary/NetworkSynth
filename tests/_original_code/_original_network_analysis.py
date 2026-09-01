# SPDX-License-Identifier: GPL-3.0-or-later
import math
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import networkit as nk
import networkx as nx
import numpy as np
import scipy.stats as stats

from scipy.optimize import linprog

max_dim = []
min_dim = []
dimension = []
holder_exp = []
widths = []
avg_frac = []
avg_closeness = []
avg_degree = []
avg_clustering = []
max_al = []
min_al = []
avg_eig = []


def wnfd_nk(G, Q, weight=True, draw=False, fdigi=0):
    N_list = []
    r_g_all_set = set()
    num_nodes_all = nx.number_of_nodes(G)
    G = nx.convert_node_labels_to_integers(G)
    if weight == True:
        G_nk = nk.nxadapter.nx2nk(G, weightAttr="weight")
    else:
        G_nk = nk.nxadapter.nx2nk(G)
    for node in G.nodes():
        grow = []
        grow_ori = (
            nk.distance.Dijkstra(G_nk, node, storePaths=False).run().getDistances()
        )
        for s in grow_ori:
            if s > 0 and s < 99999:
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
    Nw_mat = np.ones(
        (len(N_list), len(r_g_all))
    )

    for i, num in enumerate(N_list):
        for j, r in enumerate(r_g_all):
            Nw_mat[i, j] += sum(count for radius, count in num.items() if radius <= r)

    diameter = r_g_all[-1]

    Zq_list = []

    for q in Q:
        Zq_mat = np.power(Nw_mat / Nw_mat[:, -1, None], q)
        Zq_list.append(np.sum(Zq_mat, axis=0))

    tau_list = []
    if draw == True:
        plt.figure(figsize=(7, 7))
        for idx, q in enumerate(Q):
            r_g_all_np = np.array(r_g_all)
            x = np.log(r_g_all_np / diameter)
            y = np.log(Zq_list[idx])
            q = format(q, ".0f")
            plt.plot(x, y, "*", label="q=" + str(q))
            slope, intercept, _, _, _ = stats.linregress(x, y)
            tau_list.append(slope)
            plt.xlabel("ln(r/d)")
            plt.ylabel("ln(sum function)")
    else:
        for idx, q in enumerate(Q):
            r_g_all_np = np.array(r_g_all)
            x = np.log(r_g_all_np / diameter)
            y = np.log(Zq_list[idx])
            slope, intercept, _, _, _ = stats.linregress(x, y)
            tau_list.append(slope)

    return tau_list, r_g_all, diameter, Zq_list


def nspectrum(tau_list, q_list, k, color):
    al_list = []
    fal_list = []
    for i in range(1, len(q_list)):
        al = (tau_list[i] - tau_list[i - 1]) / (q_list[i] - q_list[i - 1])
        al_list.append(al)
    for j in range(len(q_list) - 1):
        fal = q_list[j] * al_list[j] - tau_list[j]
        fal_list.append(fal)
    plt.plot(al_list, fal_list, label=k, linewidth=3, color=color)
    plt.xlabel("Lipschitz-Hölder exponent, " r"$\alpha$")
    plt.ylabel("Multi-fractal spectrum, " r"$f(\alpha)$")
    alpha_0 = al_list[np.argmax(fal_list)]
    width = np.max(al_list) - np.min(al_list)
    print("Holder Exponent:", alpha_0)
    print("width:", width)
    holder_exp.append(alpha_0)
    widths.append(width)
    max_al.append(np.max(al_list))
    min_al.append(np.min(al_list))
    return alpha_0, width


def ndimension(tau_list, q_list, k, color):
    dim_list = []
    qd_list = []
    for i in range(len(q_list)):
        if q_list[i] != 0:
            dim = tau_list[i] / q_list[i]
            dim_list.append(dim)
            qd_list.append(q_list[i])

    plt.plot(qd_list, dim_list, label=k, linewidth=3, color=color)
    plt.xlabel("Distorting exponent, " r"$q$")
    plt.ylabel("Generalized fractal dimension, " r"$D(q)$")
    print("Dim_max: ", np.max(dim_list))
    print("Dim_min: ", np.min(dim_list))
    print("Dim_max-min: ", np.max(dim_list) - np.min(dim_list))
    max_dim.append(np.max(dim_list))
    min_dim.append(np.min(dim_list))
    dimension.append(np.max(dim_list) - np.min(dim_list))
    return dim_list, qd_list


def node_dimension(G, weight=True, fdigi=2):
    G = nx.convert_node_labels_to_integers(G)
    if weight:
        for u, v, d in G.edges(data=True):
            if d["weight"] != 0:
                d["weight"] = 1.0 / d["weight"]
    if not weight:
        G_nk = nk.nxadapter.nx2nk(G)
    else:
        G_nk = nk.nxadapter.nx2nk(G, weightAttr="weight")

    node_dimension_ = {}
    for node in G.nodes():
        num_nodes = 0
        grow = (
            nk.distance.Dijkstra(G_nk, int(node), storePaths=False).run().getDistances()
        )
        grow.sort()
        if weight == True:
            grow = [round(d, fdigi) for d in grow if round(d, fdigi) != 0]

        r_g, num_g = [], []
        num = Counter(grow)
        for i, j in num.items():
            num_nodes += j
            if i > 0:
                r_g.append(i)
                num_g.append(num_nodes)

        x = np.log(r_g)
        y = np.log(num_g)

        if len(r_g) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            node_dimension_[node] = slope
        else:
            node_dimension_[node] = 0
    return node_dimension_


def calculate_centralities(graph, weight_flag):
    if weight_flag:
        nfd_centrality = node_dimension(graph, weight=True)
    else:
        nfd_centrality = node_dimension(graph, weight=None)
    if weight_flag:
        closeness_centrality = nx.closeness_centrality(
            graph, distance=lambda u, v, d: 1 / d["weight"]
        )
    else:
        closeness_centrality = nx.closeness_centrality(graph, distance=None)
    if weight_flag:
        degree_centrality = dict(graph.degree(weight="weight"))
    else:
        degree_centrality = dict(graph.degree(weight=None))
    if weight_flag:
        cluster_coef = nx.clustering(graph, weight="weight")
    else:
        cluster_coef = nx.clustering(graph, weight=None)
    return {
        "nfd": list(nfd_centrality.values()),
        "closeness": list(closeness_centrality.values()),
        "degree": list(degree_centrality.values()),
        "clustering": list(cluster_coef.values()),
    }


def calculate_betweenness(G_nx, weight_flag):
    if weight_flag == "True":
        G_nx = deepcopy(G_nx)
        for u, v, d in G_nx.edges(data=True):
            if d["weight"] != 0:
                d["weight"] = 1.0 / d["weight"]
            else:
                d["weight"] = float("inf")
        G = nk.nxadapter.nx2nk(G_nx, weightAttr="weight")
    else:
        G_nx = deepcopy(G_nx)
        G = nk.nxadapter.nx2nk(G_nx, weightAttr=None)
    betweenness = nk.centrality.Betweenness(G, normalized=True).run().scores()
    return betweenness


def calculate_orc(G, weight_flag):
    G = deepcopy(G)
    G = nx.convert_node_labels_to_integers(G)
    alpha = 0.5

    if weight_flag == "True":
        for u, v, d in G.edges(data=True):
            d["weight"] = 1.0 / d["weight"] if d["weight"] != 0 else float("inf")
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    else:
        dist = dict(nx.all_pairs_shortest_path_length(G))

    curvatures = []
    for u, v in G.edges():
        nbrs_u = list(G.neighbors(u))
        nbrs_v = list(G.neighbors(v))
        sup_u = [u] + nbrs_u
        sup_v = [v] + nbrs_v

        mu_u = np.empty(len(sup_u))
        if nbrs_u:
            mu_u[0] = alpha
            mu_u[1:] = (1.0 - alpha) / len(nbrs_u)
        else:
            mu_u[0] = 1.0

        mu_v = np.empty(len(sup_v))
        if nbrs_v:
            mu_v[0] = alpha
            mu_v[1:] = (1.0 - alpha) / len(nbrs_v)
        else:
            mu_v[0] = 1.0

        n_u, n_v = len(sup_u), len(sup_v)
        cost = np.array(
            [[dist[si].get(sj, float("inf")) for sj in sup_v] for si in sup_u]
        )

        nm = n_u * n_v
        row_idx, col_idx, data = [], [], []
        for i in range(n_u):
            for j in range(n_v):
                row_idx.append(i)
                col_idx.append(i * n_v + j)
                data.append(1.0)
        for j in range(n_v):
            for i in range(n_u):
                row_idx.append(n_u + j)
                col_idx.append(i * n_v + j)
                data.append(1.0)

        from scipy import sparse as sp

        A_eq = sp.csc_matrix(
            (data, (row_idx, col_idx)), shape=(n_u + n_v, nm)
        )
        b_eq = np.concatenate([mu_u, mu_v])
        res = linprog(
            cost.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs"
        )
        w1 = res.fun if res.success else float("nan")

        d_uv = dist[u][v]
        kappa = 1.0 - w1 / d_uv if d_uv > 0 else 0.0
        curvatures.append(kappa)

    return curvatures


def calculate_eigenvector_centrality(G, weight_flag):
    if weight_flag == "True":
        eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    else:
        eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight=None)
    return list(eigen_centrality.values())


def calculate_diameter(G, weight_flag):
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = max(
            nx.diameter(G.subgraph(c).copy()) for c in nx.connected_components(G)
        )
    return diameter


def calculate_assortativity(G, weight_flag):
    if weight_flag == "True":
        assortativity_coef = nx.degree_pearson_correlation_coefficient(
            G, weight="weight"
        )
    else:
        assortativity_coef = nx.degree_pearson_correlation_coefficient(G, weight=None)
    return assortativity_coef
