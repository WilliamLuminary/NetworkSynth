# src/configs/gui_config.py
"""Config for a GUI-driven run, built from a JSON run-spec.

Lives at the ``configs/`` root rather than in a ``*_mode`` package because it is
not a mode: it is mode-*parameterised*, resolved from the spec at run time.

The spec is written by whoever launches us — StructuralGT's controller, or a
person testing by hand — and names the input files, the output directory, the
mode, and the generation parameters.  See ``INTEGRATION_PLAN.md`` section 3.

``from_spec`` returns a fresh subclass rather than mutating this class, so two
specs handled by one interpreter cannot collide.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from .base_config import BaseConfig
from .enums import DatasetId

logger = logging.getLogger(__name__)

#: Bumped when the spec shape changes incompatibly.  A spec declaring a
#: different version is rejected rather than half-understood.
SPEC_CONTRACT_VERSION = 1

#: Params given as JSON arrays that the pipelines expect as tuples.
_TUPLE_PARAMS = frozenset(
    {
        "IMAGE_SIZE",
        "FRAME_SIZE",
        "SYNTHETIC_FRAME_SIZE",
        "TILE_FRAME_SIZE",
        "TARGET_SCALE",
    }
)

_REQUIRED_KEYS = ("mode", "output_dir", "inputs", "params")


class SpecError(ValueError):
    """The run-spec is missing something or says something we cannot honour."""


class GuiConfig(BaseConfig):
    """Base for GUI runs.  Use :meth:`from_spec`; do not use this directly."""

    MODE: str = ""
    #: Input file paths from the spec.
    PATHS: Dict[str, Any] = {}

    @classmethod
    def from_spec(cls, spec_path: str) -> type:
        """Build a run config from *spec_path*.  Returns a new subclass."""
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

        version = spec.get("contract", SPEC_CONTRACT_VERSION)
        if version != SPEC_CONTRACT_VERSION:
            raise SpecError(
                f"{spec_path}: contract version {version}, but this build "
                f"understands {SPEC_CONTRACT_VERSION}"
            )

        inputs = spec["inputs"]
        for key in ("edge_list", "positions"):
            if not inputs.get(key):
                raise SpecError(f"{spec_path}: inputs.{key} is required")

        config = type("GuiRunConfig", (cls,), {})
        config.MODE = spec["mode"]
        config.PATHS = inputs
        config.BASE_OUTPUT_PATH = spec["output_dir"]
        config.DATASETS = [DatasetId(spec.get("run_name", "gui_run"))]

        for key, value in spec["params"].items():
            if key in _TUPLE_PARAMS and isinstance(value, list):
                value = tuple(value)
            setattr(config, key, value)

        logger.info(
            f"Run-spec loaded: mode={config.MODE} "
            f"output_dir={config.BASE_OUTPUT_PATH} "
            f"params={sorted(spec['params'])}"
        )
        return config

    @classmethod
    def initialize(cls):
        super().initialize()
        # Set explicitly: BaseConfig derives OUTPUT_DENOTE from a `*_mode`
        # segment in the module path, and this config has none.
        cls.OUTPUT_DENOTE = f"gui_{cls.MODE}"
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId):
        """Read the graph StructuralGT exported."""
        from graphs import read_graph_csv

        return read_graph_csv(cls.PATHS["edge_list"], cls.PATHS["positions"])

    @classmethod
    def load_original_image(cls, dataset_id: DatasetId):
        """Optional background image; absent is normal."""
        path = cls.PATHS.get("image")
        if not path:
            return None
        if not os.path.exists(path):
            logger.warning(f"Image named in the run-spec is missing: {path}")
            return None

        import cv2

        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
