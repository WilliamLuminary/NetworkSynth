import os
from datetime import datetime

from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


def calculate_edge_lengths(graph):
    return [euclidean(graph.nodes[u]['pos'], graph.nodes[v]['pos']) for u, v in graph.edges()]


def assign_weights_to_edges(graph, map_length_to_weight, length_bins, weight_baskets, y):
    edge_lengths = calculate_edge_lengths(graph)
    weights = [map_length_to_weight(length, length_bins, weight_baskets, y) for length in edge_lengths]

    for (u, v), weight in zip(graph.edges(), weights):
        graph[u][v]['weight'] = weight

    return graph


def plot_graph_with_positions(graph, set_name, resolution, title, frame, save=False,
                              base_path='/content/drive/MyDrive/vis/Results Yaxing/',
                              background=False):
    plt.figure(figsize=(10, 10))

    for s, e in graph.edges():
        x1, y1 = graph.nodes[s]['pos']
        x2, y2 = graph.nodes[e]['pos']
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=3, zorder=2)

    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        plt.plot(x, y, 'bo', markersize=3.5, zorder=2)

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
    if title:
        plt.title(title)
    plt.tight_layout()

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_folder_name = f"Set {set_name} Res {resolution}"
        synthetic_folder_name = f"Synthetic Graph - Set {set_name} Res {resolution}"
        main_folder_path = os.path.join(base_path, main_folder_name)
        synthetic_folder_path = os.path.join(main_folder_path, synthetic_folder_name)
        os.makedirs(synthetic_folder_path, exist_ok=True)
        filename = f"{title} Time {timestamp}.png"
        filepath = os.path.join(synthetic_folder_path, filename)
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def process_weighted_graph(graph, frame, map_length_to_weight, length_bins, weight_baskets, y, title, save=False,
                           base_path='/content/drive/MyDrive/vis/Results Yaxing/', skip_plotting=False):
    graph = graph.copy()
    weighted_graph = assign_weights_to_edges(graph, map_length_to_weight, length_bins, weight_baskets, y)
    if skip_plotting:
        return weighted_graph

    plot_graph_with_positions(weighted_graph, title, frame, save, base_path)

    return weighted_graph
