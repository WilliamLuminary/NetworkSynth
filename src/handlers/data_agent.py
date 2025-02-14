# src/data/data_agent.py
import logging
import os
import pickle
from typing import List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np

from analysis import MultifractalAnalyzer
from analysis_main import MultifractalBatchProcessor
from config import Config, DataType, Resolution, SetName
from graph import GraphAttrAgent
from utils import build_graph_pos_and_adj_mat, plot_network
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

        else:
            if not set_name or not resolution:
                raise ValueError("Both set_name and resolution are required for traditional initialization")
            self.set_name = set_name
            self.resolution = resolution

            self.original_network = None
            self.synthetic_networks = []

            self.saver = Saver(set_name=set_name, resolution=resolution)
            self.batch_processor = None

        self.original_image = None

        self.original_analysis = None
        self.synthetic_analysis = None

        self.spectra_image = None
        self.dimensions_image = None

        self.attributes: Optional[GraphAttrAgent] = None
        self.mapper: Optional[Mapper] = None

    def prepare_data(self):
        self.original_image, self.original_network = DataLoader(set_name=self.set_name,
                                                                resolution=self.resolution).load()
        self.attributes = GraphAttrAgent(self.original_network).analyze()
        self.mapper = Mapper(self.original_network)

    def multifractal_analysis(self):
        assert self.synthetic_networks, "Add synthetic networks at first."
        self.batch_processor = MultifractalBatchProcessor([self.original_network], self.synthetic_networks)
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
            for (image_name, image) in self.batch_processor.images:
                self.saver.save_file(image, data_type, image_name)
