import os
import pickle
from collections import Counter

import math
import matplotlib.pyplot as plt
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from tqdm import tqdm

# noinspection PyUnresolvedReferences
import main  ## Very important


# =============================================================================
# Global Configuration
# =============================================================================

def configure_plot_settings():
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.linewidth': 2,
        'axes.labelpad': 10.0,
        'axes.xmargin': 0,
        'axes.ymargin': 0,
        'xtick.major.size': 7,
        'xtick.minor.size': 3.5,
        'xtick.major.width': 1.1,
        'xtick.minor.width': 1.1,
        'ytick.major.size': 7,
        'ytick.minor.size': 3.5,
        'ytick.major.width': 1.1,
        'ytick.minor.width': 1.1,
        'lines.markersize': 10,
        'lines.markeredgewidth': 0.8,
        'figure.figsize': (9, 7),
        'figure.subplot.left': 0.15,
        'figure.subplot.right': 0.95,
        'figure.subplot.bottom': 0.13,
        'figure.subplot.top': 0.95
    })


configure_plot_settings()


# =============================================================================
# Data Loading and Setup
# =============================================================================

def load_pkl_files(base_dir):
    data_ = {}
    for root_, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pkl"):
                current_dir = os.path.basename(root_)
                if current_dir != 'synthetic':
                    continue  # Skip files not in the 'synthetic' directory

                set_name_ = os.path.basename(os.path.dirname(root_))
                file_path = os.path.join(root_, file)
                with open(file_path, 'rb') as f:
                    pkl_content = pickle.load(f)
                if set_name_ in data_:
                    print(f"Duplicate entry for {set_name_}, skipping.")
                else:
                    data_[set_name_] = pkl_content
    return data_


def get_script_dir():
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        return os.getcwd()


def wnfd_network(graph, Q, weight=True, draw=False, f_digi=0):
    ## Find radius
    N_list = []
    r_g_all_set = set()
    graph = nx.convert_node_labels_to_integers(graph)
    G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight') if weight else nk.nxadapter.nx2nk(graph)

    for node in graph.nodes():
        distances = nk.distance.Dijkstra(G_nk, node, storePaths=False).run().getDistances()
        grow = [d for d in distances if 0 < d < 99999]
        grow.sort()
        # upf = (1/pow(10,fdigi+1)*5)
        # grow = [round(d+upf,fdigi) for d in grow]
        grow = [math.ceil(d) for d in grow] if f_digi == 0 else [round(d, f_digi) for d in grow if
                                                                 round(d, f_digi) != 0]
        #         grow = grow[1:]
        num = Counter(grow)
        r_g_all_set.update(num.keys())
        N_list.append(num)

    r_g_all = np.array(sorted(list(r_g_all_set)))
    Nw_mat = np.ones((len(N_list), len(r_g_all)))  # Num_r matrix: column: node, row: radius

    for i, num in enumerate(N_list):
        for j, r in enumerate(r_g_all):
            Nw_mat[i, j] += sum(count for radius, count in num.items() if radius <= r)
    # for i, j in product(range(len(N_list)), range(len(r_g_all))):
    #     Nw_mat[i, j] += sum(count for radius, count in N_list[i].items() if radius <= r_g_all[j])

    ## Distortion factor q: get Zq_mat
    diameter = r_g_all[-1]
    Zq_list = []
    for q in Q:
        Zq_mat = np.power(Nw_mat / Nw_mat[:, -1, None], q)
        Zq_list.append(np.sum(Zq_mat, axis=0))

    ## Get tau(slope)
    tau_list = []
    for idx, q in enumerate(Q):
        x = np.log(r_g_all / diameter)
        y = np.log(Zq_list[idx])
        slope, _, _, _, _ = stats.linregress(x, y)
        tau_list.append(slope)
        if draw:
            plt.plot(x, y, '*', label=f'q={q:.0f}')
            plt.xlabel('ln(r/d)')
            plt.ylabel('ln(sum function)')
    return tau_list


def compute_wndf(G_list, weight_flag):
    Q = [q / 100 for q in range(-2000, 2001, 10)]
    wei_ntauls_list = []
    for G in G_list:
        if weight_flag == 'True':
            ntau = wnfd_network(G, Q, weight=True, draw=False, f_digi=1)  # Peiyu: weight=None
        else:
            ntau = wnfd_network(G, Q, weight=False, draw=False)
        wei_ntauls_list.append(ntau)
    return Q, wei_ntauls_list


def n_spectrum(tau_list, q_list, label: str, color):
    al_list = [(tau_list[i] - tau_list[i - 1]) / (q_list[i] - q_list[i - 1])
               for i in range(1, len(q_list))]
    fal_list = [q_list[i] * al_list[i] - tau_list[i] for i in range(len(al_list))]

    plt.plot(al_list, fal_list, label=label, linewidth=3, color=color)
    plt.xlabel('Lipschitz-Hölder exponent, 'r'$\alpha$')
    plt.ylabel('Multi-fractal spectrum, 'r'$f(\alpha)$')
    # plt.legend()
    # plt.savefig('/Users/xiongyex/Downloads'+'/Spec_{}.png'.format(k),bbox_inches = 'tight',dpi=600)
    alpha_0 = al_list[np.argmax(fal_list)]
    width = np.max(al_list) - np.min(al_list)
    max_al_val = np.max(al_list)
    min_al_val = np.min(al_list)
    print('Holder Exponent:', alpha_0)
    print('width:', width)
    return alpha_0, width, max_al_val, min_al_val


def n_dimension(tau_list, q_list, label, color):
    valid_pairs = [(q, tau) for q, tau in zip(q_list, tau_list) if q != 0]
    valid_q, valid_tau = zip(*valid_pairs)
    dim_list = [tau / q for q, tau in zip(valid_q, valid_tau)]

    plt.plot(valid_q, dim_list, label=label, linewidth=3, color=color)
    plt.xlabel('Distorting exponent, 'r'$q$')
    plt.ylabel('Generalized fractal dimension, 'r'$D(q)$')
    # plt.legend()
    # plt.savefig('/Users/xiongyex/Downloads'+'/Spec_{}.png'.format(k),bbox_inches = 'tight',dpi=600)
    dim_max = np.max(dim_list)
    dim_min = np.min(dim_list)
    diff = dim_max - dim_min
    print('Dim_max:', dim_max)
    print('Dim_min:', dim_min)
    print('Dim_max - Dim_min:', diff)
    return dim_list, dim_max, dim_min, diff


def save_violin_data(metric_list, filename, columns):
    df_ = pd.DataFrame(metric_list).T
    df_.columns = columns
    df_.to_csv(filename, index=False)


def node_dimension(graph, weight=True, fdigi=2):
    node_dimensions = {}
    graph = nx.convert_node_labels_to_integers(graph)
    if weight:
        for u, v, d in graph.edges(data=True):
            if d.get('weight', 0) != 0:
                d['weight'] = 1.0 / d['weight']
    G_nk = nk.nxadapter.nx2nk(graph, weightAttr='weight') if weight else nk.nxadapter.nx2nk(graph)

    for node in graph.nodes():
        distances = nk.distance.Dijkstra(G_nk, node, storePaths=False).run().getDistances()
        distances.sort()
        if weight:
            distances = [round(d, fdigi) for d in distances if round(d, fdigi) != 0]
        count = Counter(distances)
        num_nodes = 0
        r_g, num_g = [], []
        for dist, cnt in count.items():
            num_nodes += cnt
            if dist > 0:
                r_g.append(dist)
                num_g.append(num_nodes)
        if len(r_g) > 2:
            slope, _, _, _, _ = stats.linregress(np.log(r_g), np.log(num_g))
            node_dimensions[node] = slope
        else:
            node_dimensions[node] = 0
    return node_dimensions


def load_data():
    script_dir = get_script_dir()
    result_name = 'results_full_can_use'
    base_directory = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'output', result_name))
    pkl_data = load_pkl_files(base_directory)

    sorted_keys = sorted(pkl_data.keys(), key=lambda x: int(x))
    G_list, name = [], []
    for set_name in sorted_keys:
        sublist = pkl_data[set_name]
        if len(sublist) < 100:
            raise Exception(f"Set {set_name} has only {len(sublist)} items instead of 100.")
        sublist = sublist[:100]
        G_list.extend(sublist)
        name.extend([set_name] * 100)

    metrics = {
        'max_dim': [],
        'min_dim': [],
        'dimension': [],
        'holder_exp': [],
        'widths': [],
        'max_al': [],
        'min_al': []
    }
    return script_dir, G_list, name, metrics


def compute_centralities(G_list, weight_flag):
    nfd_centrality_list = []
    avg_frac = []
    closeness_centrality_list = []
    avg_closeness = []
    degree_centrality_list = []
    avg_degree = []
    cluster_coef_list = []
    avg_clustering = []

    for G in G_list:
        if weight_flag == 'True':
            nfd_centrality = node_dimension(G, weight=True).values()
            closeness_centrality = nx.closeness_centrality(G, distance=lambda u, v, d: 1 / d['weight']).values()
            degree_centrality = dict(G.degree(weight='weight')).values()
            cluster_coef = nx.clustering(G, weight='weight').values()

        else:
            nfd_centrality = node_dimension(G, weight=None).values()  # Peiyu: weight=None
            closeness_centrality = nx.closeness_centrality(G, distance=None).values()
            degree_centrality = dict(G.degree(weight=None)).values()
            cluster_coef = nx.clustering(G, weight=None).values()

        nfd_centrality_list.append(list(nfd_centrality))
        avg_frac.append(np.mean(np.array(nfd_centrality_list[-1])))
        closeness_centrality_list.append(list(closeness_centrality))
        avg_closeness.append(np.mean(np.array(closeness_centrality_list[-1])))
        degree_centrality_list.append(list(degree_centrality))
        avg_degree.append(np.mean(np.array(degree_centrality_list[-1])))
        cluster_coef_list.append(list(cluster_coef))
        avg_clustering.append(np.mean(np.array(cluster_coef_list[-1])))

        return (nfd_centrality_list, avg_frac,
                closeness_centrality_list, avg_closeness,
                degree_centrality_list, avg_degree,
                cluster_coef_list, avg_clustering)


def compute_betweenness(G_list, weight_flag):
    betweenness_list, avg_betweenness_list = [], []
    for G in G_list:
        if weight_flag == 'True':
            G_tmp = G.copy()
            for u, v, d in G_tmp.edges(data=True):
                d['weight'] = 1.0 / d['weight'] if d.get('weight', 0) != 0 else float('inf')  # 确保权重不为0, 若为 0，则设为无穷大
            G_nk = nk.nxadapter.nx2nk(G, weightAttr='weight')
        else:
            G_nk = nk.nxadapter.nx2nk(G, weightAttr=None)
        bt = nk.centrality.Betweenness(G_nk, normalized=True).run().scores()
        betweenness_list.append(bt)
        avg_betweenness_list.append(sum(bt) / G.number_of_nodes())
    return betweenness_list, avg_betweenness_list


def compute_orc(G_list, weight_flag):
    orc_list, avg_orcs = [], []
    for G in G_list:
        G = G.copy()
        if weight_flag == 'True':
            for u, v, d in G.edges(data=True):
                d['weight'] = 1.0 / d['weight']
            orc = OllivierRicci(nx.convert_node_labels_to_integers(G), alpha=0.5, verbose="ERROR", weight='weight')
        else:
            orc = OllivierRicci(nx.convert_node_labels_to_integers(G), alpha=0.5, verbose="ERROR", weight=None)
        orc.compute_ricci_curvature()
        curvatures = [d['ricciCurvature'] for u, v, d in orc.G.edges(data=True)]
        orc_list.append(curvatures)
        avg_orcs.append(np.mean(curvatures))
    return orc_list, avg_orcs


def compute_assortativity(G_list, weight_flag):
    assortativity_coef_list, avg_assortativity = [], []
    weight_ = 'weight' if weight_flag == 'True' else None
    for G in G_list:
        assortativity_coef = nx.degree_pearson_correlation_coefficient(G, weight=weight_)
        avg_assortativity.append(np.mean(np.array(assortativity_coef)))

    return assortativity_coef_list, avg_assortativity


def compute_eigenvector(G_list, weight_flag):
    eigen_centrality_list, avg_eigen_centrality = [], []
    for G in G_list:
        if not nx.is_connected(G):
            raise ValueError("Graph must contain only the largest connected component.")
        try:
            weight_ = 'weight' if weight_flag == 'True' else None
            eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight=weight_)
            eigen_values = list(eigen_centrality.values())
            eigen_centrality_list.append(eigen_values)
            avg_eigen_centrality.append(np.mean(eigen_values))
        except nx.PowerIterationFailedConvergence:
            print("Skipping a graph due to eigenvector centrality convergence failure.")
            eigen_centrality_list.append([np.nan] * G.number_of_nodes())  # Maintain alignment
            avg_eigen_centrality.append(np.nan)  # Keep lists aligned
    return eigen_centrality_list, avg_eigen_centrality


##################################################### save data #####################################################

def compute_diameter(G_list, weight_flag):
    if weight_flag != 'True':
        return None
    diameter_list = []
    for G in G_list:
        if nx.is_connected(G):
            diameter_list.append(nx.diameter(G))
        else:
            diameter_list.append(max(nx.diameter(G.subgraph(c).copy()) for c in nx.connected_components(G)))
            # 对于不连通的图，计算每个连通分量的直径，并取最大值
    return diameter_list


def main():
    script_dir, G_list, names, metrics = load_data()
    output_root = os.path.join(script_dir, "anal")

    kX_index = '20kX'
    weight_flag = 'False'

    Q, wei_ntauls_list = compute_wndf(G_list, weight_flag)
    np.save(os.path.join(output_root, f'div_ntauls_{kX_index}_{weight_flag}.npy'),
            wei_ntauls_list)

    # -------------------------------
    # Plot multifractal spectrum
    # -------------------------------
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    num_colors = len(wei_ntauls_list)
    cmap = plt.get_cmap('RdBu')
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    for i, ntau in enumerate(wei_ntauls_list):
        arg1 = ntau[:80]  # use first 80 elements for plotting
        arg2 = Q[:80]
        alpha, width, max_al_val, min_al_val = n_spectrum(arg1, arg2, label=names[i], color=colors[i])
        metrics['holder_exp'].append(alpha)
        metrics['widths'].append(width)
        metrics['max_al'].append(max_al_val)
        metrics['min_al'].append(min_al_val)

    ax.legend(loc='upper right')
    from matplotlib.ticker import AutoMinorLocator
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.grid(False)
    plt.savefig(os.path.join(output_root, f'weighted_{kX_index}_nspectrum.svg'), bbox_inches='tight', dpi=600)
    plt.close()

    # y_min, y_max = ax.get_ylim()
    # ax.set_ylim(y_min, y_max + 0.1)

    # -------------------------------
    # Plot generalized fractal dimension
    # -------------------------------
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(figsize=(9, 7))
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Reload previously saved wnfd data
    wei_ntauls_list = np.load(os.path.join(output_root, f'div_ntauls_{kX_index}_{weight_flag}.npy'),
                              allow_pickle=True)
    for i, ntau in enumerate(wei_ntauls_list):
        _, dim_max, dim_min, diff = n_dimension(ntau, Q, label=names[i], color=colors[i])
        metrics['max_dim'].append(dim_max)
        metrics['min_dim'].append(dim_min)
        metrics['dimension'].append(diff)

    ax.legend(loc='upper right')
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    plt.grid(False)
    plt.savefig(os.path.join(output_root, f'weighted_{kX_index}_ndimension.svg'),
                bbox_inches='tight', dpi=600)
    plt.close()

    # -------------------------------
    # Plot generalized fractal dimension
    # -------------------------------
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(figsize=(9, 7))
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Reload previously saved wnfd data
    wei_ntauls_list = np.load(os.path.join(output_root, f'div_ntauls_{kX_index}_{weight_flag}.npy'),
                              allow_pickle=True)
    for i, ntau in enumerate(wei_ntauls_list):
        _, dim_max, dim_min, diff = n_dimension(ntau, Q, label=names[i], color=colors[i])
        metrics['max_dim'].append(dim_max)
        metrics['min_dim'].append(dim_min)
        metrics['dimension'].append(diff)

    ax.legend(loc='upper right')
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    plt.grid(False)
    plt.savefig(os.path.join(output_root, f'weighted_{kX_index}_ndimension.svg'),
                bbox_inches='tight', dpi=600)
    plt.close()

    # -------------------------------
    # Compute centrality measures and save CSV files
    # -------------------------------
    (nfd_list, avg_frac,
     closeness_list, avg_closeness,
     degree_list, avg_degree,
     clustering_list, avg_clustering) = compute_centralities(G_list, weight_flag)

    save_violin_data(nfd_list, os.path.join(output_root, f'{kX_index}_nfd_violin.csv'), names)
    save_violin_data(closeness_list, os.path.join(output_root, f'{kX_index}_closeness_violin.csv'), names)
    save_violin_data(degree_list, os.path.join(output_root, f'{kX_index}_degree_violin.csv'), names)
    save_violin_data(clustering_list, os.path.join(output_root, f'{kX_index}_clustering_violin.csv'), names)

    betweenness_list, avg_betweenness_list = compute_betweenness(G_list, weight_flag)

    orc_list, avg_orcs = compute_orc(G_list, weight_flag)
    save_violin_data(orc_list, os.path.join(output_root, f'{kX_index}_orc_violin.csv'), names)

    assortativity_coef_list, avg_assortativity = compute_assortativity(G_list, weight_flag)

    diameter_list, avg_diameter = compute_diameter(G_list, weight_flag)

    eigen_list, avg_eigen = compute_eigenvector(G_list, weight_flag)
    print("Average Eigenvector Centrality:", avg_eigen)

    # -------------------------------
    # Save summary data to CSV
    # -------------------------------
    summary_data = {
        'holder_exp': metrics['holder_exp'],
        'widths': metrics['widths'],
        'dimension': metrics['dimension'],
        'avg_frac': avg_frac,
        'avg_closeness': avg_closeness,
        'avg_degree': avg_degree,
        'avg_clustering': avg_clustering,
        'avg_betweenness_list': avg_betweenness_list,
        'avg_orcs': avg_orcs,
        'avg_assortativity': avg_assortativity,
        'avg_eigen_centrality': avg_eigen,
        'max_al': metrics['max_al'],
        'min_al': metrics['min_al'],
    }
    if weight_flag == 'True' and diameter_list is not None:
        summary_data['diameter'] = diameter_list
    df_summary = pd.DataFrame(summary_data, index=names).transpose()
    output_csv = os.path.join(output_root,
                              f'{kX_index}_output.csv' if weight_flag == 'True'
                              else f'{kX_index}_uw_output.csv')
    df_summary.to_csv(output_csv)
    print("Summary data saved to", output_csv)


if __name__ == '__main__':
    main()
