from typing import List, Optional, Union

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
        self._original_path = None
        self._synthetic_path = None

        self.__original_image = None

        self.__original_network = None
        self.__synthetic_networks = []

        if mode == Mode.GEN:
            if "dataset_id" not in kwargs:
                raise ValueError("dataset_id is required for GEN mode")
            self._dataset_id = kwargs["dataset_id"]
        elif mode == Mode.ANA:
            # Two independent paths, not one directory with an assumed layout:
            # what to compare is the caller's choice.
            for key in ("original_path", "synthetic_path"):
                assert key in kwargs, f"{key} is required for ANA mode"
            self._original_path = kwargs["original_path"]
            self._synthetic_path = kwargs["synthetic_path"]

        self.__mode = mode

    def get_original_image(self) -> np.ndarray:
        return self.__original_image

    def get_original_network(self) -> Union[SynthGraph, List[SynthGraph]]:
        return self.__original_network

    def get_synthetic_networks(self) -> List[SynthGraph]:
        return self.__synthetic_networks

    def add_synthetic_graph(self, graph: SynthGraph) -> None:
        self.__synthetic_networks.append(graph)

    def load(self) -> None:
        if self.__mode == Mode.GEN:
            self.__original_network = self._load_original_network()
            self.__original_image = self._load_original_image()

        elif self.__mode == Mode.ANA:
            self.__original_network = self._config.NETWORKS_FUNC(self._original_path)
            self.__synthetic_networks = self._config.NETWORKS_FUNC(self._synthetic_path)

    def _load_original_image(self):
        return self._config.ORIGINAL_IMAGE_FUNC(self._dataset_id)

    def _load_original_network(self):
        return self._config.ORIGINAL_NETWORK_FUNC(self._dataset_id)
