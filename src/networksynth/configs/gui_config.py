# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import copyreg
import json
import logging
import os
from typing import Any, Dict, Optional

from numpy import ndarray

from networksynth.graphs.synth_graph import SynthGraph

from .base_config import BaseConfig
from .dataset_id import DatasetId
from .file_definitions import (
    INPLACE_DIR,
    ORIGINAL_DIR,
    SYNTHETIC_DIR,
    SaveSpec,
    save_network_csv,
    save_network_graphml,
    save_png,
    save_svg,
    save_webp,
)

logger = logging.getLogger(__name__)

SPEC_CONTRACT_VERSION = 2

_PAIR_OR_DIRECTORY = (
    ("edge_list", "positions"),
    ("datasets_dir",),
    ("adjacency", "positions_npy"),
    ("network_graphml",),
)
MODE_INPUTS = {
    "generate": _PAIR_OR_DIRECTORY,
    "hybrid": _PAIR_OR_DIRECTORY,
    "sweep": _PAIR_OR_DIRECTORY,
}

_EDGE_SUFFIX = "_edgelist.csv"
_POSITIONS_SUFFIX = "_positions.csv"
# What StructuralGT's exporter writes. Its columns are already the ones
# read_graph_csv accepts, so only the file names differ.
_SGT_EDGE_SUFFIX = "_EdgeList.csv"
_SGT_POSITIONS_SUFFIX = "_NodePositions.csv"
_IMAGE_SUFFIX = "_image.tif"
_MATRIX_SUFFIX = "_adjacency.npy"
_NPY_POSITIONS_SUFFIX = "_positions.npy"
_GRAPHML_SUFFIX = "_network.graphml"
_GRAPHML_GZ_SUFFIX = "_network.graphml.gz"


_DIRECTORY_FORMS = (
    (_EDGE_SUFFIX, (_POSITIONS_SUFFIX,)),
    (_SGT_EDGE_SUFFIX, (_SGT_POSITIONS_SUFFIX,)),
    (_MATRIX_SUFFIX, (_NPY_POSITIONS_SUFFIX,)),
    (_GRAPHML_SUFFIX, ()),
    (_GRAPHML_GZ_SUFFIX, ()),
)

_TUPLE_PARAMS = frozenset(
    {
        "IMAGE_SIZE",
        "FRAME_SIZE",
        "SYNTHETIC_FRAME_SIZE",
        "TILE_FRAME_SIZE",
        "TARGET_SCALE",
        "NF_RANGE",
        "EF_RANGE",
    }
)

_REQUIRED_KEYS = ("contract", "mode", "output_dir", "inputs", "params", "run_name")


class SpecError(ValueError):
    pass


NETWORK_FORMATS = {
    "csv": save_network_csv,
    "graphml": save_network_graphml,
    "graphml.gz": save_network_graphml,
}
PLOT_FORMATS = {
    "webp": save_webp,
    "png": save_png,
    "svg": save_svg,
}

_FORMATTED_OUTPUTS = {
    "network": (
        ("SAVE_ORIGINAL_NETWORK", ORIGINAL_DIR, "original_network"),
        ("SAVE_SYNTHETIC_NETWORK", SYNTHETIC_DIR, "synthetic_network"),
    ),
    "plot": (
        ("SAVE_ORIGINAL_GRAPH", ORIGINAL_DIR, "original_graph"),
        ("SAVE_SYNTHETIC_GRAPH", SYNTHETIC_DIR, "synthetic_graph"),
        ("SAVE_ANALYSIS_FIGURE", INPLACE_DIR, "analysis_figure"),
    ),
}


def format_param(group: str, name: str) -> str:
    """The switch for one output format.

    The format's name is its file extension, so it can carry a dot
    (``graphml.gz``); an attribute name cannot.
    """
    return f"WRITE_{group.upper()}_{name.upper().replace('.', '_')}"


def _rebuild_from_spec(spec_path: str) -> type:
    return GuiConfig.from_spec(spec_path)


class _SpecConfigMeta(type):
    pass


def _reduce_spec_config(cls):
    spec_path = cls.__dict__.get("SPEC_PATH")
    if spec_path is None:
        return cls.__qualname__
    return (_rebuild_from_spec, (spec_path,))


# Through copyreg, not __reduce__: pickle checks the dispatch table first, and
# once it sees a custom metaclass it saves the class by name — which a spawned
# child cannot resolve, since this config was built at run time from a spec.
copyreg.pickle(_SpecConfigMeta, _reduce_spec_config)


def discover_datasets(directory: str) -> list:
    if not os.path.isdir(directory):
        raise SpecError(f"not a directory: {directory}")

    found = {}
    for entry in sorted(os.listdir(directory)):
        for lead, partners in _DIRECTORY_FORMS:
            if not entry.endswith(lead):
                continue
            name = entry[: -len(lead)]
            if name in found:
                raise SpecError(
                    f"{directory}: '{name}' is named as two datasets at once "
                    f"({found[name]} and {lead}). Rename one of them."
                )
            for partner in partners:
                beside = os.path.join(directory, f"{name}{partner}")
                if not os.path.exists(beside):
                    raise SpecError(
                        f"{os.path.join(directory, entry)} has no {partner} "
                        f"file beside it ({beside})"
                    )
            found[name] = lead

    if not found:
        forms = ", ".join(f"*{lead}" for lead, _ in _DIRECTORY_FORMS)
        raise SpecError(f"no {forms} file found in {directory}")
    return sorted(found)


def _load_npy_pair(positions_path: str, adjacency_path: str) -> SynthGraph:
    import numpy as np

    from networksynth.utils import build_graph, transpose_positions

    positions = np.load(positions_path, allow_pickle=True)
    matrix = np.load(adjacency_path, allow_pickle=True).item()
    graph = build_graph(positions, matrix)
    transpose_positions(graph)
    return graph


def _load_single_network(path: str) -> SynthGraph:
    from networksynth.graphs import load_graphs

    graphs = load_graphs(path)
    if len(graphs) > 1:
        logger.warning(
            f"{path} holds {len(graphs)} networks; reading the first. Point at "
            "a single-network file to choose a different one."
        )
    return graphs[0]


def _load_from_directory(directory: str, name: str) -> SynthGraph:
    """One dataset out of a directory, read by the form its name is in."""
    path = os.path.join(directory, name)
    for edges, positions in (
        (_EDGE_SUFFIX, _POSITIONS_SUFFIX),
        (_SGT_EDGE_SUFFIX, _SGT_POSITIONS_SUFFIX),
    ):
        if os.path.exists(f"{path}{edges}"):
            from networksynth.graphs import read_graph_csv

            return read_graph_csv(f"{path}{edges}", f"{path}{positions}")
    if os.path.exists(f"{path}{_MATRIX_SUFFIX}"):
        return _load_npy_pair(
            f"{path}{_NPY_POSITIONS_SUFFIX}", f"{path}{_MATRIX_SUFFIX}"
        )
    if os.path.exists(f"{path}{_GRAPHML_GZ_SUFFIX}"):
        return _load_single_network(f"{path}{_GRAPHML_GZ_SUFFIX}")
    return _load_single_network(f"{path}{_GRAPHML_SUFFIX}")


class GuiConfig(BaseConfig, metaclass=_SpecConfigMeta):
    MODE: str = ""
    PATHS: Dict[str, Any] = {}

    INPUT_ORIENTATION: str = "none"

    TILE_FRAME_SIZE = None
    DATASET_FACTORS: dict = {}

    @classmethod
    def from_spec(cls, spec_path: str) -> type:
        if not os.path.exists(spec_path):
            raise SpecError(f"run-spec not found: {spec_path}")
        try:
            with open(spec_path) as handle:
                spec = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{spec_path}: not valid JSON ({exc})") from exc

        missing = [k for k in _REQUIRED_KEYS if k not in spec]
        if missing:
            raise SpecError(f"{spec_path}: missing required key(s): {missing}")

        version = spec["contract"]
        if version != SPEC_CONTRACT_VERSION:
            raise SpecError(
                f"{spec_path}: contract version {version}, but this build "
                f"understands {SPEC_CONTRACT_VERSION}"
            )

        mode = spec["mode"]
        if mode not in MODE_INPUTS:
            raise SpecError(
                f"{spec_path}: unknown mode {mode!r}; "
                f"this build understands: {sorted(MODE_INPUTS)}"
            )

        inputs = spec["inputs"]
        satisfied = [
            shape
            for shape in MODE_INPUTS[mode]
            if all(inputs.get(key) for key in shape)
        ]
        if len(satisfied) != 1:
            shapes = " or ".join(
                "{" + ", ".join(shape) + "}" for shape in MODE_INPUTS[mode]
            )
            problem = "matches more than one of" if satisfied else "fills none of"
            raise SpecError(
                f"{spec_path}: mode {mode!r} {problem} its input sets: {shapes}"
            )

        # type(cls), not type: the subclass must keep the metaclass that makes
        config = type(cls)("GuiRunConfig", (cls,), {"SPEC_PATH": spec_path})
        config.MODE = mode
        config.PATHS = inputs
        config.BASE_OUTPUT_PATH = spec["output_dir"]
        if inputs.get("datasets_dir"):
            config.DATASETS = [
                DatasetId(name) for name in discover_datasets(inputs["datasets_dir"])
            ]
        else:
            config.DATASETS = [DatasetId(spec["run_name"])]

        for key, value in spec["params"].items():
            if key in _TUPLE_PARAMS and isinstance(value, list):
                value = tuple(value)
            setattr(config, key, value)

        config._apply_output_formats()
        config._apply_snapshots()

        logger.info(
            f"Run-spec loaded: mode={config.MODE} "
            f"output_dir={config.BASE_OUTPUT_PATH} "
            f"params={sorted(spec['params'])}"
        )
        return config

    @classmethod
    def _apply_output_formats(cls) -> None:
        for group, formats in (("network", NETWORK_FORMATS), ("plot", PLOT_FORMATS)):
            named = [
                name for name in formats if hasattr(cls, format_param(group, name))
            ]
            if not named:
                continue

            chosen = [name for name in named if getattr(cls, format_param(group, name))]

            for attribute, directory, detail in _FORMATTED_OUTPUTS[group]:
                setattr(
                    cls,
                    attribute,
                    tuple(
                        SaveSpec(directory, detail, name, formats[name])
                        for name in chosen
                    ),
                )
            if chosen:
                logger.info(f"{group} outputs will be written as: {', '.join(chosen)}")
            else:
                logger.info(f"No {group} files will be written: every format is off.")

    @classmethod
    def _apply_snapshots(cls) -> None:
        if hasattr(cls, "WRITE_SNAPSHOTS") and not cls.WRITE_SNAPSHOTS:
            cls.SNAPSHOT_INTERVAL = 0

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.OUTPUT_DENOTE = f"gui_{cls.MODE}"
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId) -> SynthGraph:
        from networksynth.utils import orient_positions

        directory = cls.PATHS.get("datasets_dir")
        if directory:
            graph = _load_from_directory(directory, str(dataset_id))
        elif cls.PATHS.get("network_graphml"):
            graph = _load_single_network(cls.PATHS["network_graphml"])
        elif cls.PATHS.get("adjacency"):
            graph = _load_npy_pair(cls.PATHS["positions_npy"], cls.PATHS["adjacency"])
        else:
            from networksynth.graphs import read_graph_csv

            graph = read_graph_csv(cls.PATHS["edge_list"], cls.PATHS["positions"])

        orient_positions(graph, cls.INPUT_ORIENTATION)

        if not cls.FRAME_SIZE and not cls._has_background():
            # No image to take the window from, so the network's own extent is all
            # there is. It understates the window when nodes stop short of an edge,
            # which is why an image is preferred when one exists.
            import math

            positions = graph.positions()
            cls.FRAME_SIZE = (
                max(1, math.ceil(positions[:, 0].max())),
                max(1, math.ceil(positions[:, 1].max())),
            )
            if not cls.SYNTHETIC_FRAME_SIZE:
                cls.SYNTHETIC_FRAME_SIZE = cls.FRAME_SIZE

        return graph

    @classmethod
    def _has_background(cls) -> bool:
        directory = cls.PATHS.get("datasets_dir")
        if directory:
            return True
        return bool(cls.PATHS.get("image"))

    @classmethod
    def load_original_image(cls, dataset_id: DatasetId) -> Optional[ndarray]:
        directory = cls.PATHS.get("datasets_dir")
        if directory:
            candidate = os.path.join(directory, f"{dataset_id}{_IMAGE_SUFFIX}")
            path = candidate if os.path.exists(candidate) else None
        else:
            path = cls.PATHS.get("image")
        if not path:
            return None
        if not os.path.exists(path):
            logger.warning(f"Image named in the run-spec is missing: {path}")
            return None

        import cv2

        from networksynth.utils import resize_image

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Image could not be read: {path}")
            return None

        cls.IMAGE_SIZE = (image.shape[0], image.shape[1])
        if not cls.FRAME_SIZE:
            # Node coordinates are in the source image's pixels, so its own size is
            # the coordinate window. Left unset, that is the answer, and the image
            # needs no scaling to match it.
            cls.FRAME_SIZE = (image.shape[1], image.shape[0])
            if not cls.SYNTHETIC_FRAME_SIZE:
                cls.SYNTHETIC_FRAME_SIZE = cls.FRAME_SIZE
            return image
        return resize_image(image, cls.FRAME_SIZE)
