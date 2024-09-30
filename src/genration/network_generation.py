import inspect
import os
import warnings
from collections import deque
from datetime import datetime

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.genration.GraphNode import reset_graph_node, GraphNode
from src.utils.debug_utils import debugging, timer


def within_frame(position, frame):
    x, y = position
    return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]


@debugging
def calculate_frame(center_x, center_y, frame_range):
    half_range = frame_range / 2
    stack = inspect.stack()

    if stack[1].function == 'wrapper':
        caller = stack[2].function
    else:
        caller = stack[1].function

    frame = [
        [round(center_x - half_range, 2), round(center_x + half_range, 2)],
        [round(center_y - half_range, 2), round(center_y + half_range, 2)]
    ]
    print(f"{caller}: Frame {frame}", debug=True)
    return frame


def generate_nodes_and_edges(frame_range, **kwargs):
    # Reset GraphNode
    reset_graph_node(**kwargs)
    root_node = GraphNode((0, 0))
    node_set, edge_set = {root_node}, set()
    frame = calculate_frame(root_node.position[0], root_node.position[1], frame_range)

    def bfs(root):
        node_queue = deque([root])
        while node_queue:
            if len(node_set) >= 1e5:
                warnings.warn(f"Excessive node count detected: {len(node_set)}. Aborting...")
                break

            current_node = node_queue.popleft()
            if not within_frame(current_node.position, frame):
                # print(f"[DEBUG] Out of frame")
                continue

            if current_node.generate_children():
                # print(f"[DEBUG] {current_node}, Degree:{current_node.degree}, Children: {current_node.children}")
                for child in current_node.children:
                    if child == current_node:
                        continue  # Skip the parent node
                    node_set.add(child)
                    edge_set.add((current_node.position, child.position))
                    node_queue.append(child)  # Warning: Potential infinite loop due to merging

    bfs(root_node)
    return node_set, edge_set


def filter_graph(graph, frame):
    nodes_to_keep = {node for node, pos in nx.get_node_attributes(graph, 'pos').items() if within_frame(pos, frame)}
    filtered_graph = graph.subgraph(nodes_to_keep).copy()

    if len(filtered_graph.nodes) > 0:
        largest_cc = max(nx.connected_components(filtered_graph), key=len)
        filtered_graph = filtered_graph.subgraph(largest_cc).copy()

    return filtered_graph


def build_graph_from_nodes_and_edges(nodes, edges):
    graph = nx.Graph()
    position_map = {node.position: node for node in nodes}
    for node in nodes:
        graph.add_node(position_map[node.position].id, pos=node.position)

    for edge in edges:
        u, v = position_map[edge[0]].id, position_map[edge[1]].id
        graph.add_edge(u, v)
    return graph


@timer
@debugging
def plot_and_generate_network(frame_range: int = 510, regenerate_times: int = 100, skip_plotting=False, save_plot=False,
                              plot_in_frame=True, base_path='/content/drive/MyDrive/vis/Results Yaxing',
                              background=False, **kwargs):
    if regenerate_times <= 0:
        warnings.warn(
            "Both node_set and edge_set are None. regenerate_times must be greater than 0. Setting regenerate_times to 100.")
        regenerate_times = 100

    raw_nodes, raw_edges = set(), set()
    reset_graph_node(if_init=True, **kwargs)
    for _ in range(regenerate_times):
        if len(raw_nodes) > 100:
            print(
                f"Nodes generated: {len(raw_nodes)}, edges: {len(raw_edges)}.\tAborted edges: {GraphNode.aborted_edge}, merged edges: {GraphNode.merged_edge}",
                debug=True)
            break

        raw_nodes, raw_edges = generate_nodes_and_edges(
            # frame_range=frame_range*(10*avg_length / frame_range)
            frame_range=frame_range * 1.2
        )
    else:
        raise Exception(f"Too many attempts to generate graphs. Abort...")

    raw_graph = build_graph_from_nodes_and_edges(raw_nodes, raw_edges)
    raw_positions = np.array(list(nx.get_node_attributes(raw_graph, 'pos').values()))
    center_x, center_y = raw_positions[:, 0].mean(), raw_positions[:, 1].mean()
    # print(f"x: {np.min(raw_positions[:, 0])} ~ {np.max(raw_positions[:, 0])}; y: {np.min(raw_positions[:, 1])} ~ {np.max(raw_positions[:, 1])}", debug=True)
    frame = calculate_frame(center_x, center_y, frame_range)
    # frame = calculate_frame(0, 0, frame_range)
    synthetic_graph = filter_graph(raw_graph, frame)

    if skip_plotting:
        print(f"In-frame nodes: {len(synthetic_graph.nodes())}, edges: {len(synthetic_graph.edges())}", debug=True)
        return synthetic_graph, frame

    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_in_frame:
        ax.set_xlim(frame[0])
        ax.set_ylim(frame[1])

    positions = nx.get_node_attributes(synthetic_graph, 'pos')
    for u, v in synthetic_graph.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, zorder=2)

    for node, pos in positions.items():
        ax.plot(pos[0], pos[1], 'bo', markersize=2.5, zorder=2)

    scale = False  # Disabled by default
    if scale:
        scale_length = 100
        scale_x_start = frame[0][0] + 50
        scale_y = frame[1][0] + 50
        background_rect = plt.Rectangle((scale_x_start - 5, scale_y - 10), scale_length + 10, 20, facecolor='white',
                                        edgecolor='none', zorder=3)
        ax.add_patch(background_rect)
        ax.plot([scale_x_start, scale_x_start + scale_length], [scale_y, scale_y], 'k-', linewidth=2, zorder=3)
        ax.text(scale_x_start + scale_length / 2, scale_y + 5, f'{scale_length}', ha='center', fontsize=12,
                color='black', zorder=4)

    background_rect = plt.Rectangle((frame[0][0], frame[1][0]),
                                    frame[0][1] - frame[0][0], frame[1][1] - frame[1][0],
                                    facecolor=(0, 0, 0, 0.3), edgecolor='none',
                                    zorder=1) if background else plt.Rectangle((frame[0][0], frame[1][0]),
                                                                               frame[0][1] - frame[0][0],
                                                                               frame[1][1] - frame[1][0],
                                                                               facecolor='none',
                                                                               edgecolor=(0, 0, 0, 0.8), linewidth=2,
                                                                               zorder=1)
    ax.add_patch(background_rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if save_plot:
        graph_title = kwargs.get('graph_title', "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{graph_title}_graph_{timestamp}.png'
        plt.savefig(os.path.join(base_path, filename), dpi=600)
        print(f"Graph saved as {filename}")
        plt.close()
    else:
        plt.show()

    print(f"In-frame nodes: {len(synthetic_graph.nodes())}, edges: {len(synthetic_graph.edges())}", debug=True)
    return synthetic_graph, frame
