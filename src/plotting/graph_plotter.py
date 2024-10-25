# src/plotting/graph_plotter.py


from utils.base import BaseConfig
from utils.plotting_utils import plot_graph


class GraphPlotter(BaseConfig):
    def __init__(self, config):
        super().__init__(config)

    def plot_graph_with_positions(self, graph, frame, set_name='', resolution='', title='', save=False,
                                  background=False):
        plot_graph(
            graph=graph,
            set_name=set_name,
            resolution=resolution,
            title=title,
            frame=frame,
            save=save,
            base_path=self.config.OUTPUT_DIR,
            background=background,
            linewidth=2,
            node_size=2.5
        )
