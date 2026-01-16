from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from config import BaseConfig, DatasetId, Mode


class DataLoader:
    def __init__(self, mode, **kwargs):
        self._dataset_id: Optional[DatasetId] = None
        self._both_networks_path = None

        self.__original_image = None

        self.__original_network = None
        self.__synthetic_networks = []

        if mode == Mode.GEN:
            if "dataset_id" not in kwargs:
                raise ValueError("dataset_id is required for GEN mode")
            self._dataset_id = kwargs["dataset_id"]
        elif mode == Mode.ANA:
            assert "path" in kwargs, "path is required."
            self._both_networks_path = kwargs["path"]
        elif mode == Mode.ATR:
            assert "path" in kwargs, "path is required."
            self._attr_path = kwargs["path"]
            self.__attr = None

        self.__mode = mode

    def get_original_image(self) -> np.ndarray:
        return self.__original_image

    def get_original_network(self) -> Union[nx.Graph, List[nx.Graph]]:
        return self.__original_network

    def get_attr_dict(self) -> Dict:
        return self.__attr

    def get_synthetic_networks(self) -> List[nx.Graph]:
        return self.__synthetic_networks

    def add_synthetic_graph(self, graph: nx.Graph) -> None:
        self.__synthetic_networks.append(graph)

    def load(self) -> None:
        if self.__mode == Mode.GEN:
            self.__original_network = self._load_original_network()
            self.__original_image = self._load_original_image()

        elif self.__mode == Mode.ANA:
            self.__original_network, self.__synthetic_networks = (
                self._load_both_networks()
            )

        elif self.__mode == Mode.ATR:
            self.__attr = self._load_attr()

    def _load_original_image(self):
        return BaseConfig.ORIGINAL_IMAGE_FUNC(self._dataset_id)

    def _load_original_network(self):
        return BaseConfig.ORIGINAL_NETWORK_FUNC(self._dataset_id)

    def _load_both_networks(self) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        return BaseConfig.NETWORKS_FUNC(self._both_networks_path)

    def _load_attr(self) -> Dict:
        return BaseConfig.ATTRIBUTES_DICT_FUNC(self._attr_path)
