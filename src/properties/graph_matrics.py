from collections import Counter

import networkx as nx
import numpy as np
from rich import Console

from src.preprocess.raw_files_process import load_and_plot_graph

console = Console()
graph_results = {}


def load_or_generate_graph(set_name, resolution, base_path='/content/drive/MyDrive/vis'):
    cache_key = f"{set_name}_{resolution}"
    if cache_key not in graph_results:
        graph = load_and_plot_graph(set_name, resolution, save_path=base_path, load_image=False, save=False)
        return graph, "Original"
    else:
        return None, None


def prepare_graph_metrics(graph) -> dict:
    degree_count = dict(graph.degree())
    total_nodes = graph.number_of_nodes()
    degree_distribution = {k: v / total_nodes for k, v in Counter(degree_count.values()).items()}  # Keep as the fraction

    mean_num_nodes = total_nodes
    mean_num_edges = graph.number_of_edges()
    avg_degree = np.mean(list(degree_count.values()))
    avg_closeness_centrality = np.mean(list(nx.closeness_centrality(graph).values()))

    graph_metrics = {
        "degree_distribution": degree_distribution,
        "mean_num_nodes": mean_num_nodes,
        "mean_num_edges": mean_num_edges,
        "avg_closeness_centrality": avg_closeness_centrality,
        "avg_degree": avg_degree
    }
    return graph_metrics


def calculate_graph_metrics(set_name=None, resolution=None, base_path='/content/drive/MyDrive/vis', graph=None):
    if graph is None:
        if set_name is None or resolution is None:
            raise ValueError("Either 'G' or both 'set_name' and 'resolution' must be provided.")

        graph, mode = load_or_generate_graph(set_name, resolution, base_path)
        if graph is None:
            raise ValueError("Graph could not be loaded or created.")
    else:
        mode = "Synthetic"

    return prepare_graph_metrics(graph), mode


def display_degree_distribution(d: dict, color='red'):
    max_degree_len = max(len(str(degree)) for degree in d)
    for degree, fraction in sorted(d.items()):
        percentage = fraction * 100  # Convert to percentage for display
        console.print(f"- Degree {degree:>{max_degree_len}}: {round(percentage, 2):>6}%", style=color)
    return max_degree_len


def display_graph_metrics(graph_metrics, set_name, resolution, mode):
    mode_mapping = {
        "Original": (
            "green", f"In [bold]Original[/bold] Graph [bold]{set_name} - {resolution}[/bold], the distribution is:"),
        "Synthetic": (
            "red", f"In [bold]Synthetic[/bold] Graph [bold]{set_name} - {resolution}[/bold], the distribution is:")
    }

    if mode not in mode_mapping:
        raise ValueError("Invalid mode. Available modes: 'Original', 'Synthetic'.")

    color, header = mode_mapping[mode]
    console.print(header, style=color)

    max_degree_len = display_degree_distribution(graph_metrics["degree_distribution"], color)

    console.print(f"Mean no. nodes: {graph_metrics['mean_num_nodes']}", style=color)
    console.print(f"Mean no. edges: {graph_metrics['mean_num_edges']}", style=color)
    console.print(f"Avg degree: {graph_metrics['avg_degree']:>{max_degree_len + 8}.4f}", style=color)
    console.print(f"Avg closeness centrality: {graph_metrics['avg_closeness_centrality']:.4f}", style=color)

    console.print(f"Mode used: [bold]{mode}[/bold]", style=color)


def process_and_display_graph_metrics(set_name=None, resolution=None, graph=None, base_path='/content/drive/MyDrive/vis',
                                      force_update=False):
    cache_key = f"{set_name}_{resolution}" if set_name and resolution else None

    mode = "Original"
    if cache_key and cache_key in graph_results and not force_update:
        graph_metrics = graph_results[cache_key]
        graph, _ = load_or_generate_graph(set_name, resolution, base_path) if graph is None else (graph, "Synthetic")
    else:
        graph_metrics, mode = calculate_graph_metrics(graph=graph, set_name=set_name, resolution=resolution,
                                                      base_path=base_path)
        if cache_key:
            graph_results[cache_key] = graph_metrics

    if graph_metrics:
        display_graph_metrics(graph_metrics, set_name, resolution, mode)

    return graph_metrics, graph
