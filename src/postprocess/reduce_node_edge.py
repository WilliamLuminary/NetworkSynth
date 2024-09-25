import os
import warnings
from datetime import datetime

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.utils.debug_utils import debugging


def keep_largest_connected_component(graph):
    if graph.number_of_nodes() == 0:
        return graph
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    return graph


@debugging
def adjust_degree_distribution(graph, ori_graph_metrics: dict, enable_closeness_centrality: bool = False):
    target_num_nodes = ori_graph_metrics['mean_num_nodes']
    target_num_edges = ori_graph_metrics['mean_num_edges']
    target_avg_deg = ori_graph_metrics['avg_degree']
    target_avg_closeness_centrality = ori_graph_metrics['avg_closeness_centrality']

    graph = graph.copy()

    while 2 * graph.number_of_edges() / graph.number_of_nodes() > 1.1 * target_avg_deg:
        if enable_closeness_centrality:
            current_avg_closeness_centrality = np.mean(list(nx.closeness_centrality(graph).values()))
            if current_avg_closeness_centrality <= target_avg_closeness_centrality:
                break

        if graph.number_of_nodes() < 0.8 * target_num_nodes:
            warnings.warn(
                f"Low Nodes. Current: N: {graph.number_of_nodes()} E: {graph.number_of_edges()}, Target: {target_num_nodes}")
            break

        highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]
        print(f"Reducing degree of node {highest_degree_node}, current degree: {graph.degree[highest_degree_node]}",
              debug=True)

        neighbors = list(graph.neighbors(highest_degree_node))
        if neighbors:
            graph.remove_edge(highest_degree_node, neighbors[0])

        print(f"Remaining nodes: {graph.number_of_nodes()}, Remaining edges: {graph.number_of_edges()}", debug=True)

    graph = keep_largest_connected_component(graph)

    if graph.number_of_nodes() > target_num_nodes:
        print(f"High Nodes. Current: N: {graph.number_of_nodes()} E: {graph.number_of_edges()}, TN: {target_num_edges}",
              debug=True)

    if graph.number_of_edges() > target_num_edges:
        print(f"High Edges. Current: N: {graph.number_of_nodes()} E: {graph.number_of_edges()}, TE: {target_num_edges}",
              debug=True)
    print(f"AVG DEG: {2 * graph.number_of_edges() / graph.number_of_nodes()}", debug=True)
    return graph


def plot_reduced_graph(reduced_graph, title, set_name, resolution, save, frame,
                       base_path='/content/drive/MyDrive/vis/Results Yaxing/', background=False):
    plt.figure(figsize=(10, 10))

    positions = nx.get_node_attributes(reduced_graph, 'pos')
    filtered_nodes = list(reduced_graph.nodes)
    filtered_edges = list(reduced_graph.edges)

    for s, e in filtered_edges:
        pos_s = positions.get(s)
        pos_e = positions.get(e)
        if pos_s and pos_e:
            plt.plot([pos_s[0], pos_e[0]], [pos_s[1], pos_e[1]], 'r-', linewidth=3, zorder=2)

    for node in filtered_nodes:
        pos = positions.get(node)
        if pos:
            plt.plot(pos[0], pos[1], 'bo', markersize=3.5, zorder=2)

    if frame is not None:
        plt.xlim(frame[0])
        plt.ylim(frame[1])

    background_rect = plt.Rectangle((frame[0][0], frame[1][0]),
                                    frame[0][1] - frame[0][0], frame[1][1] - frame[1][0],
                                    facecolor=(0, 0, 0, 0.3), edgecolor='none',
                                    zorder=1) if background else plt.Rectangle((frame[0][0], frame[1][0]),
                                                                               frame[0][1] - frame[0][0],
                                                                               frame[1][1] - frame[1][0],
                                                                               facecolor='none',
                                                                               edgecolor=(0, 0, 0, 0.8), linewidth=2,
                                                                               zorder=1)
    plt.gca().add_patch(background_rect)

    plt.axis('off')
    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"Synthetic Graph - Set {set_name} Res {resolution}"
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        filename = f"Synthetic Graph - Set {set_name} Res {resolution} Time {timestamp}.png"
        filepath = os.path.join(folder_path, filename)
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

    # Calculate crucial graph metrics
    degrees = list(dict(reduced_graph.degree()).values())
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    degree_distribution = dict(zip(unique_degrees, counts / sum(counts)))

    mean_num_nodes = reduced_graph.number_of_nodes()
    mean_num_edges = reduced_graph.number_of_edges()
    avg_degree = np.mean(degrees)
    avg_closeness_centrality = np.mean(list(nx.closeness_centrality(reduced_graph).values()))

    return {
        "degree_distribution": degree_distribution,
        "mean_num_nodes": mean_num_nodes,
        "mean_num_edges": mean_num_edges,
        "avg_closeness_centrality": avg_closeness_centrality,
        "avg_degree": avg_degree
    }
