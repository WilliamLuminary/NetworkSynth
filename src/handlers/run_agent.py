# src/data/run_agent.py
import inspect
import logging
from typing import Optional

import networkx as nx
from matplotlib import pyplot as plt
from numpy import ndarray

from analysis import MultifractalBatchProcessor
from config import BaseConfig, DataType, FILE_CONFIGURATIONS, FileTag, Mode, Resolution, SetName
from utils import calculate_frame, finalize_plot
from .attributes_calculator import AttributesCalculator
from .data_loader import DataLoader
from .mapper import Mapper
from .saver import Saver

logger = logging.getLogger(__name__)


class RunAgent:
    def __init__(self,
                 *,
                 set_name: SetName = None,
                 resolution: Resolution = None,
                 networks_path: Optional[str] = None,
                 attr_path: Optional[str] = None):
        if networks_path:
            self.mode = Mode.ANA
            self.data_loader = DataLoader(Mode.ANA, path=networks_path)
            self.saver = Saver(output_dir=networks_path)
            self.batch_processor = None
        elif set_name and resolution:
            self.mode = Mode.GEN
            self.data_loader = DataLoader(Mode.GEN, set_name=set_name, resolution=resolution)
            self.saver = Saver(set_name=set_name, resolution=resolution)
            self.attributes: Optional[AttributesCalculator] = None
            self.mapper: Optional[Mapper] = None
            self.batch_processor = None
        elif attr_path:
            self.mode = Mode.ATR
            self.data_loader = DataLoader(Mode.ATR, path=attr_path)
            self.saver = Saver(output_dir=attr_path)
            self.attributes: Optional[AttributesCalculator] = None
            self.mapper: Optional[Mapper] = None
            self.batch_processor = None
        else:
            assert False, "Invalid arguments."

    def prepare_data(self):
        if self.mode == Mode.GEN:
            self.data_loader.load()
            original_network = self.data_loader.get_original_network()
            self.attributes = AttributesCalculator(original_network).analyze()
            self.mapper = Mapper(original_network)

        elif self.mode == Mode.ANA:
            self.data_loader.load()
            original_networks = self.data_loader.get_original_network()
            synthetic_networks = self.data_loader.get_synthetic_networks()
            self.batch_processor = MultifractalBatchProcessor(original_networks, synthetic_networks)

        elif self.mode == Mode.ATR:
            self.data_loader.load()
            self.attributes = AttributesCalculator.from_dict(self.data_loader.get_attr_dict())

    def multifractal_analysis_in_generate_mode(self):
        assert self.mode == Mode.GEN, "This method is only available in Generate mode."
        original_networks = [self.data_loader.get_original_network()]
        synthetic_networks = self.data_loader.get_synthetic_networks()
        self.batch_processor = MultifractalBatchProcessor(original_networks, synthetic_networks)
        self.multifractal_analysis()

    def multifractal_analysis(self):
        assert self.mode == Mode.ANA, "This method is only available in Analyze mode."
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
        if not self.saver and not data_type.has_tag(FileTag.PLOT):
            return

        file_name_prefix = f"{file_name_prefix}" if file_name_prefix and file_name_prefix[-1] != '_' else (
                file_name_prefix or '')

        if data_type == DataType.ORIGINAL_IMAGE:
            assert self.mode == Mode.GEN, "This data type is only available in Generate mode."
            original_image = self.data_loader.get_original_image()
            self.saver.save_file(original_image, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_NETWORK:
            assert self.mode == Mode.GEN, "This data type is only available in Generate mode."
            original_network = self.data_loader.get_original_network()
            self.saver.save_file(original_network, data_type, file_name_prefix)

        elif data_type == DataType.ORIGINAL_PROPERTY:
            assert self.mode == Mode.GEN, "This data type is only available in Generate mode."
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
            if self.saver:
                self.saver.save_file(original_figure, data_type, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_GRAPH:
            assert isinstance(arg, nx.Graph), "Must be a networkx.Graph synthetic network."
            show_figure_if_not_saving: bool = not self.saver and arg
            synthetic_figure = plot_network(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=arg,
                show=show_figure_if_not_saving
            )
            if not show_figure_if_not_saving:
                self.saver.save_file(synthetic_figure, DataType.SYNTHETIC_GRAPH, file_name_prefix)

        elif data_type == DataType.SYNTHETIC_NETWORK:
            synthetic_networks = self.data_loader.get_synthetic_networks()
            self.saver.save_file(synthetic_networks, data_type, file_name_prefix)

        elif data_type == DataType.ANALYSIS_DATA:
            assert self.mode == Mode.ANA, "This data type is only available in Analyze mode."
            self.saver.save_file({'original_multifractal_analysis_results': self.batch_processor.get_original_data(),
                                  'synthetic_multifractal_analysis_results': self.batch_processor.get_synthetic_data()},
                                 data_type,
                                 file_name_prefix)

        elif data_type == DataType.ANALYSIS_FIGURE:
            assert self.mode == Mode.ANA, "This data type is only available in Analyze mode."
            for image_name, image in self.batch_processor.get_images().items():
                self.saver.save_file(image, data_type, f"{image_name}_")

        else:
            logger.error(f"Please configure save() for {data_type}.")


def plot_network(data_type: DataType,
                 graph: nx.Graph,
                 **kwargs) -> ndarray:
    """
    :param data_type:
    :param graph: If provided, plot the graph directly.
    :param kwargs: Title, frame, plot_in_frame, background, image, alpha, edge_width, node_size, output_path
    """
    assert data_type.has_tag(FileTag.PLOT), f"{data_type} shouldn't call {inspect.currentframe().f_code.co_name}."

    file_config = FILE_CONFIGURATIONS.get(data_type)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    position_dict = nx.get_node_attributes(graph, 'pos')
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
        frame = (0, BaseConfig.DEFAULT_FRAME_SIZE[0]), (0, BaseConfig.DEFAULT_FRAME_SIZE[1])
    else:
        frame = calculate_frame(graph)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type is DataType.ORIGINAL_GRAPH:
        image = kwargs['background']
        alpha = getattr(file_config, 'alpha', 1.0)
        ax.imshow(image, cmap='gray', alpha=alpha)
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
    show_on_the_fly = kwargs.get('show', False) or getattr(file_config, 'show_on_the_fly', False)
    return finalize_plot(fig, show_on_the_fly)
