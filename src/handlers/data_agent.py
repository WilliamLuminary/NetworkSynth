# src/data/data_agent.py
import logging
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from analysis.multifractal_batch_processor import MultifractalBatchProcessor
from config import Config, DataType, FILE_CONFIGURATIONS, FileTag, Resolution, SetName
from config.enums import Mode
from graph import GraphAttrAgent
from utils import calculate_frame
from utils.utils import finalize_plot
from .data_loader import DataLoader
from .mapper import Mapper
from .saver import Saver

logger = logging.getLogger(__name__)


class DataAgent:
    def __init__(self,
                 set_name: SetName = None,
                 resolution: Resolution = None,
                 *,
                 analyze_source_path: Optional[str] = None):
        if analyze_source_path:
            self.mode = Mode.Analyze
            self.data_loader = DataLoader(analyze_source_path=analyze_source_path)
            self.saver = Saver(output_dir=analyze_source_path)
            self.batch_processor = None

        elif set_name and resolution:
            self.mode = Mode.Generate
            self.data_loader = DataLoader(set_name=set_name, resolution=resolution)
            self.saver = Saver(set_name=set_name, resolution=resolution)
            self.attributes: Optional[GraphAttrAgent] = None
            self.mapper: Optional[Mapper] = None
            self.batch_processor = None
        else:
            assert False, "Invalid arguments."

    def prepare_data(self):
        if self.mode == Mode.Generate:
            self.data_loader.load()
            original_network = self.data_loader.get_original_network()
            self.attributes = GraphAttrAgent(original_network).analyze()
            self.mapper = Mapper(original_network)

        elif self.mode == Mode.Analyze:
            self.data_loader.load()
            original_networks = [self.data_loader.get_original_network()]
            synthetic_networks = self.data_loader.get_synthetic_networks()
            self.batch_processor = MultifractalBatchProcessor(original_networks, synthetic_networks)

    def multifractal_analysis_in_generate_mode(self):
        assert self.mode == Mode.Generate, "This method is only available in Generate mode."
        original_networks = [self.data_loader.get_original_network()]
        synthetic_networks = self.data_loader.get_synthetic_networks()
        self.batch_processor = MultifractalBatchProcessor(original_networks, synthetic_networks)
        self.multifractal_analysis()

    def multifractal_analysis(self):
        assert self.mode == Mode.Analyze, "This method is only available in Analyze mode."
        assert self.batch_processor, "Batch processor is not initialized."
        self.batch_processor.process().plot()

    def add_synthetic_graph(self, graph: nx.Graph):
        """
        Add a synthetic graph to the list of synthetic graphs.
        NOT THREAD-SAFE.
        """
        self.data_loader.add_synthetic_graph(graph)

    def get_original_network(self):
        return self.data_loader.get_original_network()

    def save(self, data_type: DataType, file_name_prefix: Optional[str] = None, arg=None):
        if not self.saver and data_type != DataType.SYNTHETIC_GRAPH:
            return

        show_figure_if_not_saving: bool = not self.saver and arg
        file_name_prefix = f"{file_name_prefix}" if file_name_prefix and file_name_prefix[-1] != '_' else (
                file_name_prefix or '')

        if data_type == DataType.ORIGINAL_IMAGE:
            assert self.mode == Mode.Generate, "This data type is only available in Generate mode."
            original_image = self.data_loader.get_original_image()
            self.saver.save_file(original_image, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_NETWORK:
            assert self.mode == Mode.Generate, "This data type is only available in Generate mode."
            original_network = self.data_loader.get_original_network()
            self.saver.save_file(original_network, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_PROPERTY:
            assert self.mode == Mode.Generate, "This data type is only available in Generate mode."
            self.saver.save_file(self.attributes.savable(), data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_GRAPH:
            original_network = self.data_loader.get_original_network()
            original_image = self.data_loader.get_original_image()
            original_figure = plot_network(
                data_type=DataType.ORIGINAL_GRAPH,
                graph=original_network,
                background=original_image,
                show=True
            )
            self.saver.save_file(original_figure, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_GRAPH:
            assert isinstance(arg, nx.Graph), "Must be a networkx.Graph synthetic network."
            synthetic_figure = plot_network(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=arg,
                show=show_figure_if_not_saving
            )
            self.saver.save_file(synthetic_figure, DataType.SYNTHETIC_GRAPH, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_NETWORK:
            assert self.mode == Mode.Generate, "This data type is only available in Generate mode."
            synthetic_networks = self.data_loader.get_synthetic_networks()
            self.saver.save_file(synthetic_networks, data_type, file_name_prefix)

        elif data_type == DataType.ANALYSIS_DATA:
            assert self.mode == Mode.Analyze, "This data type is only available in Analyze mode."
            self.saver.save_file({'original_multifractal_analysis_results': self.batch_processor.original,
                                  'synthetic_multifractal_analysis_results': self.batch_processor.synthetic}, data_type,
                                 file_name_prefix)

        elif data_type == DataType.ANALYSIS_FIGURE:
            assert self.mode == Mode.Analyze, "This data type is only available in Analyze mode."
            for image_name, image in self.batch_processor.images.items():
                self.saver.save_file(image, data_type, f"{image_name}_")

        logger.error(f"Please configure save() for {data_type}.")


def plot_network(data_type: DataType,
                 graph: nx.Graph,
                 adjust_axis: bool = False, **kwargs) -> ndarray:
    """
    :param data_type:
    :param graph: If provided, plot the graph directly.
    :param adjust_axis:
    :param kwargs: Title, frame, plot_in_frame, background, image, alpha, edge_width, node_size, output_path
    """
    if not data_type.has_tag(FileTag.PLOT):
        raise ValueError(f"Invalid data type: {data_type}, only graphs can be passed to this method.")

    file_config = FILE_CONFIGURATIONS.get(data_type)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    position_dict = nx.get_node_attributes(graph, 'pos')
    if adjust_axis and position_dict is not None:
        positions_array = np.array([position_dict[node] for node in graph.nodes()])
        positions_array[:, [1, 0]] = positions_array[:, [0, 1]]
        positions_array[:, 1] = Config.DEFAULT_FRAME_SIZE - positions_array[:, 1]
        position_dict = {node: pos for node, pos in zip(graph.nodes(), positions_array)}

    edge_width = file_config.line_width
    for u, v in graph.edges():
        pos_u = position_dict.get(u)
        pos_v = position_dict.get(v)
        if pos_u is not None and pos_v is not None:
            x_values = [pos_u[0], pos_v[0]]
            y_values = [pos_u[1], pos_v[1]]
            ax.plot(x_values, y_values, 'r-', linewidth=edge_width, zorder=2)

    node_size = file_config.node_size
    for node in graph.nodes():
        pos = position_dict.get(node)
        if pos is not None:
            ax.plot(pos[0], pos[1], 'bo', markersize=node_size, zorder=2)

    if data_type is DataType.ORIGINAL_GRAPH:
        frame = (0, Config.DEFAULT_FRAME_SIZE[0]), (0, Config.DEFAULT_FRAME_SIZE[1])
    else:
        frame = calculate_frame(graph)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type is DataType.ORIGINAL_GRAPH:
        image = kwargs['background']
        alpha = getattr(file_config, 'alpha', 1.0)
        ax.imshow(image, cmap='gray', extent=(0, image.shape[0], 0, image.shape[1]), alpha=alpha)
    else:
        ax.add_patch(
            plt.Rectangle(
                (frame[0][0], frame[1][0]),
                frame[0][1] - frame[0][0],
                frame[1][1] - frame[1][0],
                facecolor='none',
                edgecolor=(0, 0, 0, 0.8),
                linewidth=2,
                zorder=1
            ))

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return finalize_plot(fig, getattr(file_config, 'show_on_the_fly', False))
