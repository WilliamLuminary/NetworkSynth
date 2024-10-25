# src/plotting/graph_plotter.py

from utils.plotting_utils import plot_graph


def plot_graph_with_positions(graph, frame, set_name='', resolution='', title='', save=False, base_path='data/Results',
                              background=False):
    plot_graph(
        graph=graph,
        set_name=set_name,
        resolution=resolution,
        title=title,
        frame=frame,
        save=save,
        base_path=base_path,
        background=background,
        linewidth=2,
        node_size=2.5
    )
