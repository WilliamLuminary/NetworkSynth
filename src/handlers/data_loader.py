from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np

from config import BaseConfig
from config.enums import Mode


class DataLoader:
    def __init__(self, mode, **kwargs):
        self._set_name = None
        self._resolution = None
        self._both_networks_path = None

        self._original_image = None

        self._original_network = None
        self._synthetic_networks = []

        if mode == Mode.GEN:
            assert 'set_name' in kwargs and 'resolution' in kwargs, "set_name and resolution are required."
            self._set_name = kwargs['set_name']
            self._resolution = kwargs['resolution']
        elif mode == Mode.ANA:
            assert 'path' in kwargs, "path is required."
            self._both_networks_path = kwargs['path']
        elif mode == Mode.ATR:
            assert 'path' in kwargs, "path is required."
            self._attr_path = kwargs['path']
            self._attr = None

        self._mode = mode

    def get_original_image(self) -> np.ndarray:
        return self._original_image

    def get_original_network(self) -> Union[nx.Graph, List[nx.Graph]]:
        return self._original_network

    def get_attr_dict(self) -> Dict:
        return self._attr

    def get_synthetic_networks(self) -> List[nx.Graph]:
        return self._synthetic_networks

    def add_synthetic_graph(self, graph: nx.Graph) -> None:
        self._synthetic_networks.append(graph)

    def load(self) -> None:
        if self._mode == Mode.GEN:
            self._original_network = self._load_original_network()
            self._original_image = self._load_original_image()

        elif self._mode == Mode.ANA:
            self._original_network, self._synthetic_networks = self._load_both_networks()

        elif self._mode == Mode.ATR:
            self._attr = self._load_attr()

    def _load_original_image(self):
        return BaseConfig.ORIGINAL_IMAGE_FUNC(str(self._set_name), str(self._resolution))

    def _load_original_network(self):
        return BaseConfig.ORIGINAL_NETWORK_FUNC(str(self._set_name), str(self._resolution))

    def _load_both_networks(self) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        return BaseConfig.NETWORKS_FUNC(self._both_networks_path)

    def _load_attr(self) -> Dict:
        return BaseConfig.ATTRIBUTES_DICT_FUNC(self._attr_path)
