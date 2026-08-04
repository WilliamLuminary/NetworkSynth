from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from configs import DatasetId, Mode
from graphs.synth_graph import SynthGraph


class DataLoader:
    def __init__(self, mode, config, **kwargs):
        """Load a dataset's inputs.

        *config* supplies the loader functions (``ORIGINAL_NETWORK_FUNC`` and
        friends).  It is passed in rather than read from the global
        ``BaseConfig`` so the active config is explicit.
        """
        self._config = config
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

    def get_original_network(self) -> Union[SynthGraph, List[SynthGraph]]:
        return self.__original_network

    def get_attr_dict(self) -> Dict:
        return self.__attr

    def get_synthetic_networks(self) -> List[SynthGraph]:
        return self.__synthetic_networks

    def add_synthetic_graph(self, graph: SynthGraph) -> None:
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
        return self._config.ORIGINAL_IMAGE_FUNC(self._dataset_id)

    def _load_original_network(self):
        return self._config.ORIGINAL_NETWORK_FUNC(self._dataset_id)

    def _load_both_networks(self) -> Tuple[List[SynthGraph], List[SynthGraph]]:
        return self._config.NETWORKS_FUNC(self._both_networks_path)

    def _load_attr(self) -> Dict:
        return self._config.ATTRIBUTES_DICT_FUNC(self._attr_path)
