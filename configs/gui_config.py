from __future__ import annotations

import copyreg
import json
import logging
import os
from typing import Any, Dict

from .base_config import BaseConfig
from .enums import DatasetId

logger = logging.getLogger(__name__)

#: Bumped when the spec shape changes incompatibly.  A spec declaring a
#: different version is rejected rather than half-understood.
SPEC_CONTRACT_VERSION = 2

#: The input shapes each mode accepts, in order — a mode does not read one
#: fixed thing: generation takes either a single network or a directory of them,
#: analysis takes a directory of finished results.  A spec must satisfy one
#: shape completely; a mode absent from here is rejected, because the
#: alternative is a spec that looks valid until a loader is handed nothing.
_PAIR_OR_DIRECTORY = (("edge_list", "positions"), ("datasets_dir",))
MODE_INPUTS = {
    "generate": _PAIR_OR_DIRECTORY,
    "hybrid": _PAIR_OR_DIRECTORY,
    "sweep": _PAIR_OR_DIRECTORY,
    "analyze": (("networks_dir",),),
}

#: The naming convention a dataset directory follows.  It is not a new
#: convention: it is what every mode already writes, so our own output re-enters
#: as input with no conversion.
_EDGE_SUFFIX = "_edgelist.csv"
_POSITIONS_SUFFIX = "_positions.csv"
_IMAGE_SUFFIX = "_image.tif"

#: Params given as JSON arrays that the pipelines expect as tuples.
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

_REQUIRED_KEYS = ("mode", "output_dir", "inputs", "params")


class SpecError(ValueError):
    pass


def _rebuild_from_spec(spec_path: str) -> type:
    return GuiConfig.from_spec(spec_path)


class _SpecConfigMeta(type):
    """Marks a config built from a run-spec, so it can be pickled.

    ``from_spec`` creates its config at run time, and pickle stores classes *by
    name* — a spawned child re-imports this module and finds no such name.  So a
    spec-built config travels as the path it came from and is rebuilt there.  The
    spec being a file on disk is what makes that possible.

    Without this, ``hybrid`` fails under a spawn start method: it hands the config
    class to a subprocess, where every other mode passes an immutable
    ``SynthParams`` instead.
    """


def _reduce_spec_config(cls):
    """How to pickle a class whose metaclass is :class:`_SpecConfigMeta`.

    Registered through ``copyreg`` rather than as ``__reduce__`` on the
    metaclass: pickle checks the ``copyreg`` dispatch table *before* it notices a
    custom metaclass, and once it does notice one it falls straight back to
    saving the class by name, ignoring ``__reduce__`` entirely.

    Returning a plain string tells pickle "resolve this by name", which is right
    for ``GuiConfig`` itself — only its spec-built subclasses need rebuilding.
    """
    spec_path = cls.__dict__.get("SPEC_PATH")
    if spec_path is None:
        return cls.__qualname__
    return (_rebuild_from_spec, (spec_path,))


copyreg.pickle(_SpecConfigMeta, _reduce_spec_config)


def discover_datasets(directory: str) -> list:
    """Every dataset in *directory*, by the convention above, sorted by name.

    An edge list with no positions file beside it stops the run and names the
    orphan: a directory silently processed minus one dataset is a wrong answer,
    not a smaller one.
    """
    if not os.path.isdir(directory):
        raise SpecError(f"not a directory: {directory}")

    names = []
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(_EDGE_SUFFIX):
            continue
        name = entry[: -len(_EDGE_SUFFIX)]
        partner = os.path.join(directory, f"{name}{_POSITIONS_SUFFIX}")
        if not os.path.exists(partner):
            raise SpecError(
                f"{os.path.join(directory, entry)} has no positions file "
                f"beside it ({partner})"
            )
        names.append(name)

    if not names:
        raise SpecError(f"no '*{_EDGE_SUFFIX}' file found in {directory}")
    return names


class GuiConfig(BaseConfig, metaclass=_SpecConfigMeta):

    MODE: str = ""
    #: Input file paths from the spec.
    PATHS: Dict[str, Any] = {}

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

        version = spec.get("contract", SPEC_CONTRACT_VERSION)
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
            # Two satisfied shapes is as wrong as none: it does not say which
            # input the run should read, and picking one silently would make a
            # caller's mistake look like a working run.
            problem = "matches more than one of" if satisfied else "fills none of"
            raise SpecError(
                f"{spec_path}: mode {mode!r} {problem} its input sets: {shapes}"
            )

        # type(cls), not type: the subclass must keep the metaclass that makes
        # it picklable.
        config = type(cls)("GuiRunConfig", (cls,), {"SPEC_PATH": spec_path})
        config.MODE = mode
        config.PATHS = inputs
        config.BASE_OUTPUT_PATH = spec["output_dir"]
        # A directory becomes one dataset per network in it, so a single run
        # produces N outputs in N subdirectories — which RunPaths and the
        # manifest already model.
        if inputs.get("datasets_dir"):
            config.DATASETS = [
                DatasetId(name) for name in discover_datasets(inputs["datasets_dir"])
            ]
        else:
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
        if cls.MODE == "analyze":
            # Reuses the analyze mode's own loader rather than a second copy:
            # what counts as a result directory is that mode's business.
            from .analyze_mode.config_sample import SampleConfig as _Ana

            cls.NETWORKS_DATA_PATH = cls.PATHS["networks_dir"]
            cls.NETWORKS_FUNC = _Ana._load_networks_dict

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId):
        from graphs import read_graph_csv

        edge_list, positions = cls._network_paths(dataset_id)
        return read_graph_csv(edge_list, positions)

    @classmethod
    def _network_paths(cls, dataset_id: DatasetId):
        directory = cls.PATHS.get("datasets_dir")
        if not directory:
            return cls.PATHS["edge_list"], cls.PATHS["positions"]
        return (
            os.path.join(directory, f"{dataset_id}{_EDGE_SUFFIX}"),
            os.path.join(directory, f"{dataset_id}{_POSITIONS_SUFFIX}"),
        )

    @classmethod
    def load_original_image(cls, dataset_id: DatasetId):
        directory = cls.PATHS.get("datasets_dir")
        if directory:
            # Optional, and the only per-dataset file that is: a network is
            # analysable without its image.
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

        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
