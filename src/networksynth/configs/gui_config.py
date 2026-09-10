# SPDX-License-Identifier: GPL-3.0-or-later
"""A config built from the JSON run-spec the window writes."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from numpy import ndarray

from networksynth.graphs.synth_graph import SynthGraph

from .base_config import BaseConfig, GenerateConfig, HybridConfig, SweepConfig
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
from .loaders import (
    IMAGE_SUFFIX,
    SGT_EDGE_SUFFIX,
    SGT_POSITIONS_SUFFIX,
    SpecError,
    discover_datasets,
    load_csv_pair,
    load_network,
    load_network_file,
    load_npy_pair,
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

# StructuralGT skeletonises a copy scaled to this on its longest side, and exports
# the untouched image beside it, so its coordinates are in the scaled copy's pixels.
_SGT_MAX_SIDE = 1024

_REQUIRED_KEYS = ("contract", "mode", "output_dir", "inputs", "params", "run_name")


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


def _load_network(paths: Dict[str, Any], dataset_id: DatasetId) -> SynthGraph:
    directory = paths.get("datasets_dir")
    if directory:
        graph, _ = load_network(directory, str(dataset_id))
        return graph
    if paths.get("network_graphml"):
        return load_network_file(paths["network_graphml"])
    if paths.get("adjacency"):
        return load_npy_pair(paths["positions_npy"], paths["adjacency"])
    return load_csv_pair(paths["edge_list"], paths["positions"])


def _image_path(paths: Dict[str, Any], dataset_id: DatasetId) -> Optional[str]:
    directory = paths.get("datasets_dir")
    if directory:
        return os.path.join(directory, f"{dataset_id}{IMAGE_SUFFIX}")
    return paths.get("image") or None


def _sgt_export(paths: Dict[str, Any], dataset_id: DatasetId) -> bool:
    directory = paths.get("datasets_dir")
    if directory:
        return os.path.exists(os.path.join(directory, f"{dataset_id}{SGT_EDGE_SUFFIX}"))
    positions = paths.get("positions") or ""
    return positions.endswith(SGT_POSITIONS_SUFFIX)


def _frame_from_inputs(
    paths: Dict[str, Any], orientation: str, dataset_id: DatasetId
) -> Tuple[int, int]:
    """The coordinate window when the spec leaves it open.

    The image's own size when there is one; otherwise the network's extent,
    which understates the window when nodes stop short of an edge.
    """
    path = _image_path(paths, dataset_id)
    if path and os.path.exists(path):
        import cv2

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SpecError(f"image could not be read: {path}")
        height, width = image.shape[:2]
        if _sgt_export(paths, dataset_id):
            longest = max(height, width)
            scale = _SGT_MAX_SIDE / longest if longest > _SGT_MAX_SIDE else 1.0
            return (round(width * scale), round(height * scale))
        return (width, height)

    from networksynth.utils import orient_positions

    graph = _load_network(paths, dataset_id)
    orient_positions(graph, orientation)
    positions = graph.positions()
    return (
        max(1, math.ceil(positions[:, 0].max())),
        max(1, math.ceil(positions[:, 1].max())),
    )


@dataclass(frozen=True, kw_only=True)
class _SpecInputs:
    PATHS: Dict[str, Any]
    INPUT_ORIENTATION: str = "none"

    def load_original_network(self, dataset_id: DatasetId) -> SynthGraph:
        from networksynth.utils import orient_positions

        graph = _load_network(self.PATHS, dataset_id)
        orient_positions(graph, self.INPUT_ORIENTATION)
        return graph

    def load_original_image(self, dataset_id: DatasetId) -> Optional[ndarray]:
        path = _image_path(self.PATHS, dataset_id)
        if not path:
            return None
        if not os.path.exists(path):
            if not self.PATHS.get("datasets_dir"):
                logger.warning(f"Image named in the run-spec is missing: {path}")
            return None

        import cv2

        from networksynth.utils import resize_image

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Image could not be read: {path}")
            return None
        return resize_image(image, self.FRAME_SIZE)


@dataclass(frozen=True, kw_only=True)
class GuiGenerateConfig(_SpecInputs, GenerateConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class GuiHybridConfig(_SpecInputs, HybridConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class GuiSweepConfig(_SpecInputs, SweepConfig):
    pass


GUI_CONFIG_CLASSES = {
    cls.MODE: cls for cls in (GuiGenerateConfig, GuiHybridConfig, GuiSweepConfig)
}


def _output_formats(params: Dict[str, Any]) -> Dict[str, tuple]:
    """SAVE_* fields from the WRITE_* switches, which are popped off params."""
    fields = {}
    for group, formats in (("network", NETWORK_FORMATS), ("plot", PLOT_FORMATS)):
        switches = {
            name: params.pop(format_param(group, name))
            for name in formats
            if format_param(group, name) in params
        }
        if not switches:
            continue
        chosen = [name for name, on in switches.items() if on]
        for attribute, directory, detail in _FORMATTED_OUTPUTS[group]:
            fields[attribute] = tuple(
                SaveSpec(directory, detail, name, formats[name]) for name in chosen
            )
        if chosen:
            logger.info(f"{group} outputs will be written as: {', '.join(chosen)}")
        else:
            logger.info(f"No {group} files will be written: every format is off.")
    return fields


def from_spec(spec_path: str) -> BaseConfig:
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
        shape for shape in MODE_INPUTS[mode] if all(inputs.get(key) for key in shape)
    ]
    if len(satisfied) != 1:
        shapes = " or ".join(
            "{" + ", ".join(shape) + "}" for shape in MODE_INPUTS[mode]
        )
        problem = "matches more than one of" if satisfied else "fills none of"
        raise SpecError(
            f"{spec_path}: mode {mode!r} {problem} its input sets: {shapes}"
        )

    if inputs.get("datasets_dir"):
        datasets = [
            DatasetId(name) for name in discover_datasets(inputs["datasets_dir"])
        ]
    else:
        datasets = [DatasetId(spec["run_name"])]

    # JSON has no tuples, so every list came from one.
    params = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in spec["params"].items()
    }
    params.update(_output_formats(params))
    if params.pop("WRITE_SNAPSHOTS", True) is False:
        params["SNAPSHOT_INTERVAL"] = 0

    if not params.get("FRAME_SIZE"):
        params["FRAME_SIZE"] = _frame_from_inputs(
            inputs, params.get("INPUT_ORIENTATION", "none"), datasets[0]
        )
    if not params.get("SYNTHETIC_FRAME_SIZE"):
        params["SYNTHETIC_FRAME_SIZE"] = params["FRAME_SIZE"]

    try:
        config = GUI_CONFIG_CLASSES[mode](
            DATASETS=datasets,
            PATHS=inputs,
            BASE_OUTPUT_PATH=spec["output_dir"],
            OUTPUT_DENOTE=f"gui_{mode}",
            **params,
        )
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{spec_path}: {exc}") from exc

    logger.info(
        f"Run-spec loaded: mode={mode} output_dir={config.BASE_OUTPUT_PATH} "
        f"params={sorted(spec['params'])}"
    )
    return config
