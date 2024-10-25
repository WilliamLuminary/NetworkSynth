# src/utils/plotting_utils.py

import datetime
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.output_handler import OutputHandler


def plot_graph(graph, positions=None, title=None, frame=None, save=False, output_path=None, background=False,
               image=None, alpha=1.0, adjust_positions=False, show=True, linewidth=2, node_size=2.5,
               plot_in_frame=True):
    fig, ax = plt.subplots(figsize=(10, 10))

    if positions is None:
        positions = nx.get_node_attributes(graph, 'pos')

    if adjust_positions:
        positions_array = np.array([positions[node] for node in graph.nodes()])
        positions_array[:, [1, 0]] = positions_array[:, [0, 1]]
        positions_array[:, 1] = 510 - positions_array[:, 1]
        positions = {node: pos for node, pos in zip(graph.nodes(), positions_array)}

    if image is not None and background:
        image_extent = [0, image.shape[1], 0, image.shape[0]]
        ax.imshow(image, cmap='gray', extent=image_extent, alpha=alpha)

    for u, v in graph.edges():
        pos_u = positions.get(u)
        pos_v = positions.get(v)
        if pos_u is not None and pos_v is not None:
            x_values = [pos_u[0], pos_v[0]]
            y_values = [pos_u[1], pos_v[1]]
            ax.plot(x_values, y_values, 'r-', linewidth=linewidth, zorder=2)

    for node in graph.nodes():
        pos = positions.get(node)
        if pos is not None:
            ax.plot(pos[0], pos[1], 'bo', markersize=node_size, zorder=2)

    if frame is not None and plot_in_frame:
        ax.set_xlim(frame[0])
        ax.set_ylim(frame[1])

    if frame is not None:
        if background:
            facecolor = (0, 0, 0, 0.3)
            edgecolor = 'none'
        else:
            facecolor = 'none'
            edgecolor = (0, 0, 0, 0.8)
        background_rect = plt.Rectangle(
            (frame[0][0], frame[1][0]),
            frame[0][1] - frame[0][0],
            frame[1][1] - frame[1][0],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
            zorder=1
        )
        ax.add_patch(background_rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    if title:
        plt.title(title)

    plt.tight_layout()

    if save:
        if output_path is None:
            raise ValueError("Output path must be specified when save=True.")

        output_handler = OutputHandler()
        output_handler.ensure_directory(output_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{title} Time {timestamp}.png" if title else f"Graph_{timestamp}.png"
        filepath = os.path.join(output_path, filename)

        # Archive existing file if it exists
        output_handler.archive_if_exists(filepath)

        plt.savefig(filepath, dpi=600)
        plt.close()
        print(f"Saved plot: {filepath}")
    else:
        if show:
            plt.show()
        else:
            plt.close()
