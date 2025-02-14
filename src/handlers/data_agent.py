# src/data/data_agent.py
import logging
import os
import pickle
from typing import List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from analysis.multifractal_batch_processor import MultifractalBatchProcessor
from config import Config, DataType, FILE_CONFIGURATIONS, FileTag, Resolution, SetName
from config.enums import Mode
from graph import GraphAttrAgent
from utils import build_graph_pos_and_adj_mat, calculate_frame
from utils.utils import finalize_plot
from .mapper import Mapper
from .saver import Saver

logger = logging.getLogger(__name__)


def _resize_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]
    scaling_factor = target_size / min(height, width)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def _trim_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]

    bottom = min(height, target_size)
    right = min(width, target_size)
    image = image[:bottom, :right]
    return image


def _transform_coordinates_by_image(network: nx.Graph, image: np.ndarray, flip_x=False, flip_y=False,
                                    rotation_deg=270):
    size = image.shape
    c = (size[0] / 2, size[1] / 2)

    if not flip_x and not flip_y and rotation_deg == 0:
        return

    for node, d in network.nodes(data=True):
        p = np.array(d.get('pos', [0, 0]), dtype=np.float64)

        p[0] -= c[0]
        p[1] -= c[1]

        if flip_x:
            p[0] = -p[0]
        if flip_y:
            p[1] = -p[1]

        if rotation_deg == 90:
            px, py = -p[1], p[0]
            p[0], p[1] = px, py
        elif rotation_deg == 180:
            p = -p
        elif rotation_deg == 270:
            px, py = p[1], -p[0]
            p[0], p[1] = px, py

        p[0] += c[0]
        p[1] += c[1]
        d['pos'] = p


class DataLoader:
    def __init__(self, **kwargs):
        if 'set_name' in kwargs and 'resolution' in kwargs:
            self.set_name = kwargs['set_name']
            self.resolution = kwargs['resolution']

            self.original_network = build_graph_pos_and_adj_mat((self._load_positions(), self._load_sparse_matrix()))
            self.original_image = self._load_image()

    def load(self):
        if not Config.DEFAULT_FRAME_SIZE:
            Config.update_frame_size((self.original_image.shape[1], self.original_image.shape[0]))

        image = _resize_cv2_image(self.original_image)
        image = _trim_cv2_image(image)
        _transform_coordinates_by_image(self.original_network, image)
        return self.original_image, self.original_network

    def _load_positions(self) -> Union[np.ndarray, List]:
        return Config.POSITION_DATA_FUNC(str(self.set_name), str(self.resolution))

    def _load_sparse_matrix(self) -> Union[np.ndarray, List]:
        return Config.ADJ_MATRIX_DATA_FUNC(str(self.set_name), str(self.resolution))

    def _load_image(self) -> Union[np.ndarray, List]:
        return Config.IMAGES_FUNC(str(self.set_name), str(self.resolution))


def _load_network_pkl(folder: str) -> List[nx.Graph]:
    for file in os.listdir(folder):
        if file.endswith('.pkl') and 'network' in file:
            with open(os.path.join(folder, file), 'rb') as f:
                content = pickle.load(f)
                return [content] if isinstance(content, nx.Graph) else content
    return []


def _load_networks(data_path: str):
    original_network, synthetic_networks = None, None
    for entry in os.listdir(data_path):
        entry_path = os.path.join(data_path, entry)
        if entry == 'synthetic':
            synthetic_networks = _load_network_pkl(entry_path)
        elif entry in ('origin', 'original'):
            original_network = _load_network_pkl(entry_path)
    return original_network, synthetic_networks


class DataAgent:
    def __init__(self,
                 set_name: SetName = None,
                 resolution: Resolution = None,
                 *,
                 analyze_source_path: Optional[str] = None):
        if analyze_source_path:
            self.set_name = None
            self.resolution = None

            self.original_network, self.synthetic_networks = _load_networks(analyze_source_path)

            self.saver = Saver(output_dir=analyze_source_path)
            self.batch_processor = MultifractalBatchProcessor(self.original_network, self.synthetic_networks)
            self.mode = Mode.Analyze
        else:
            if not set_name or not resolution:
                raise ValueError("Both set_name and resolution are required for traditional initialization")
            self.set_name = set_name
            self.resolution = resolution

            self.original_network = None
            self.synthetic_networks = []

            self.saver = Saver(set_name=set_name, resolution=resolution)
            self.batch_processor = None
            self.mode = Mode.Generate

        self.original_image = None

        self.original_analysis = None
        self.synthetic_analysis = None

        self.attributes: Optional[GraphAttrAgent] = None
        self.mapper: Optional[Mapper] = None

    def prepare_data(self):
        assert self.mode == Mode.Generate, "This method is only for generating data."
        self.original_image, self.original_network = DataLoader(set_name=self.set_name,
                                                                resolution=self.resolution).load()
        self.attributes = GraphAttrAgent(self.original_network).analyze()
        self.mapper = Mapper(self.original_network)

    def multifractal_analysis(self):
        assert self.synthetic_networks, "Add synthetic networks at first."
        network = self.original_network if isinstance(self.original_network, list) else [self.original_network]
        self.batch_processor = MultifractalBatchProcessor(network, self.synthetic_networks)
        self.batch_processor.process().plot()

    def add_synthetic_graph(self, graph: nx.Graph):
        """
        Add a synthetic graph to the list of synthetic graphs.
        NOT THREAD-SAFE.
        """
        self.synthetic_networks.append(graph)

    # def get_synthetic_graph(self):
    #     return self.synthetic_networks.get() if not self.synthetic_networks.empty() else None

    def set_attributes(self, attributes: GraphAttrAgent):
        self.attributes = attributes

    def save(self, data_type: DataType, file_name_prefix: Optional[str] = None, arg=None):
        if not self.saver and data_type != DataType.SYNTHETIC_GRAPH:
            return

        show_figure_if_not_saving: bool = not self.saver and arg

        file_name_prefix = f"{file_name_prefix}" if file_name_prefix and file_name_prefix[-1] != '_' else (
                file_name_prefix or '')
        if data_type == DataType.ORIGINAL_IMAGE:
            self.saver.save_file(self.original_image, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_NETWORK:
            self.saver.save_file(self.original_network, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_PROPERTY:
            self.saver.save_file(self.attributes, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_GRAPH:
            original_figure = plot_network(
                data_type=DataType.ORIGINAL_GRAPH,
                graph=self.original_network,
                background=self.original_image,
                show=True
            )
            self.saver.save_file(original_figure, data_type, file_name_prefix)
        elif data_type == DataType.SYNTHETIC_GRAPH:
            assert isinstance(arg, nx.Graph), \
                "Content must be a networkx.Graph object and should be a synthetic network."
            synthetic_figure = plot_network(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=arg,
                show=show_figure_if_not_saving
            )
            self.saver.save_file(synthetic_figure, DataType.SYNTHETIC_GRAPH, file_name_prefix)
            self.add_synthetic_graph(arg)
        elif data_type == DataType.SYNTHETIC_NETWORK:
            self.saver.save_file(self.synthetic_networks, data_type, file_name_prefix)
        elif data_type == DataType.ANALYSIS_DATA:
            self.saver.save_file(self.batch_processor.original, data_type, file_name_prefix)
            self.saver.save_file(self.batch_processor.synthetic, data_type, file_name_prefix)
        elif data_type == DataType.ANALYSIS_FIGURE:
            for image_name, image in self.batch_processor.images.items():
                self.saver.save_file(image, data_type, f"{image_name}_")


def plot_network(data_type: DataType,
                 graph: nx.Graph,
                 adjust_axis: bool = False, **kwargs) -> ndarray:
    """
    :param data_type:
    :param graph: If provided, plot the graph directly.
    :param pos_and_adj_mat: Only used if `graph` is not provided.
    Tuple / List / Ndarray of positions and adjacency matrix.
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
