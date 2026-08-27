from __future__ import annotations

import copyreg
import json
import logging
import os
from typing import Any, Dict, Optional

from numpy import ndarray

from graphs.synth_graph import SynthGraph

from .base_config import BaseConfig
from .dataset_id import DatasetId
from .file_definitions import (
    INPLACE_DIR,
    ORIGINAL_DIR,
    SYNTHETIC_DIR,
    SaveSpec,
    save_network_csv,
    save_network_nkbin,
    save_pickle,
    save_png,
    save_svg,
    save_webp,
)

logger = logging.getLogger(__name__)

#: Bumped when the spec shape changes incompatibly.  A spec declaring a
#: different version is rejected rather than half-understood.
SPEC_CONTRACT_VERSION = 2

#: The input shapes each mode accepts, in order — a mode does not read one
#: fixed thing: generation takes either a single network or a directory of them,
#: analysis takes a directory of finished results.  A spec must satisfy one
#: shape completely; a mode absent from here is rejected, because the
#: alternative is a spec that looks valid until a loader is handed nothing.
#: One entry per format the loaders read, in the order the form offers them.
#: ``.nkbin`` is absent deliberately: it is written, never read back here.
_PAIR_OR_DIRECTORY = (
    ("edge_list", "positions"),
    ("datasets_dir",),
    ("adjacency", "positions_npy"),
    ("network_pkl",),
)
MODE_INPUTS = {
    "generate": _PAIR_OR_DIRECTORY,
    "hybrid": _PAIR_OR_DIRECTORY,
    "sweep": _PAIR_OR_DIRECTORY,
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

# "contract" and "run_name" are required, not defaulted: defaulting the
# contract defeats the only check that catches version drift.
_REQUIRED_KEYS = ("contract", "mode", "output_dir", "inputs", "params", "run_name")


class SpecError(ValueError):
    pass


#: What the GUI can write, per kind of output, and the serialiser each needs.
#: The name is the extension.  Split in two because the serialisers are not
#: interchangeable: ``save_svg`` takes a matplotlib figure and nothing else,
#: and ``save_network_csv`` takes one graph.
NETWORK_FORMATS = {
    "csv": save_network_csv,
    "pkl": save_pickle,
    "nkbin": save_network_nkbin,
}
PLOT_FORMATS = {
    "webp": save_webp,
    "png": save_png,
    "svg": save_svg,
}

#: The outputs each group covers: ``(attribute, directory, name)``.
#:
#: Three are deliberately not here.  ``synthetic_network`` is the batch — a
#: *list* of graphs, which only a pickle can hold.  ``original_image`` is the
#: input image rather than a plot, so the vector format cannot apply.  Reports
#: and properties are text and a pickle by nature.
_FORMATTED_OUTPUTS = {
    "network": (
        ("SAVE_ORIGINAL_NETWORK", ORIGINAL_DIR, "original_network"),
        ("SAVE_SYNTHETIC_EXPORT", SYNTHETIC_DIR, "synthetic_network"),
    ),
    "plot": (
        ("SAVE_ORIGINAL_GRAPH", ORIGINAL_DIR, "original_graph"),
        ("SAVE_SYNTHETIC_GRAPH", SYNTHETIC_DIR, "synthetic_graph"),
        ("SAVE_ANALYSIS_FIGURE", INPLACE_DIR, "analysis_figure"),
    ),
}


def format_param(group: str, name: str) -> str:
    """The run-spec param that switches one format on, e.g. WRITE_NETWORK_CSV.

    Derived rather than written out twice, so the form and the config cannot
    drift into disagreeing about what a checkbox is called.
    """
    return f"WRITE_{group.upper()}_{name.upper()}"


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


def _load_npy_pair(positions_path: str, adjacency_path: str) -> SynthGraph:
    """A ``_pos.npy`` / ``_mat.npy`` pair, read the way every CLI config reads one.

    The transpose belongs to the format rather than to any one dataset: these
    files record (row, column) where the rest of the toolkit expects (x, y),
    which is why every config that loads a pair transposes it too.
    """
    import numpy as np

    from utils import build_graph, transpose_positions

    positions = np.load(positions_path, allow_pickle=True)
    # .item() unwraps the 0-d object array a scipy sparse matrix is saved as;
    # a plain dense array raises here rather than being read as the wrong thing.
    matrix = np.load(adjacency_path, allow_pickle=True).item()
    graph = build_graph(positions, matrix)
    transpose_positions(graph)
    return graph


def _load_pickled(path: str) -> SynthGraph:
    """The network in a pickle, or the first of the batch in one.

    A run writes its synthetic networks as one pickled list, so that file is
    the obvious thing to hand back in as an input.  Taking the first is said
    out loud, because which one it was is not otherwise visible.
    """
    from graphs import load_graphs

    graphs = load_graphs(path)
    if len(graphs) > 1:
        logger.warning(
            f"{path} holds {len(graphs)} networks; reading the first. Point at "
            "a single-network pickle to choose a different one."
        )
    return graphs[0]


class GuiConfig(BaseConfig, metaclass=_SpecConfigMeta):
    MODE: str = ""
    #: Input file paths from the spec.
    PATHS: Dict[str, Any] = {}

    # Hybrid geometry BaseConfig requires but the GUI form does not offer; a
    # spec can still override these through its params.  Values match
    # configs/hybrid_mode/config_sample.py.
    #: A quarter turn applied to the input network as it is read, for data
    #: recorded in a different orientation to the image it was traced from.
    #: See ``utils.orient_positions``; the image is never turned.
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
        """Rebuild the save specs from the formats the spec asked for.

        A group the spec says nothing about keeps ``BaseConfig``'s default, so
        a spec written by hand needs none of this.  A group it names and leaves
        entirely off writes nothing for those outputs, which is a real request:
        a run made only to look at the result wants as little on disk as
        possible.  The batch of synthetic networks is not in either group, so
        that still lands and a preview still has something to read.
        """
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
        """The form's switch, as the interval the pipelines actually read.

        They take one number where the form takes a switch and a number: zero
        is off, and an interval left set behind a switch turned off would take
        snapshots anyway.  A spec that says nothing about the switch is left
        alone, so a hand-written one still means what it says.
        """
        if hasattr(cls, "WRITE_SNAPSHOTS") and not cls.WRITE_SNAPSHOTS:
            cls.SNAPSHOT_INTERVAL = 0

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        # Set explicitly: BaseConfig derives OUTPUT_DENOTE from a `*_mode`
        # segment in the module path, and this config has none.
        cls.OUTPUT_DENOTE = f"gui_{cls.MODE}"
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId) -> SynthGraph:
        """The network named in the spec, read the way its format asks.

        Which keys the spec filled decide, because that is what the shape check
        in :meth:`from_spec` has already established; nothing sniffs the file.
        """
        from utils import orient_positions

        if cls.PATHS.get("network_pkl"):
            graph = _load_pickled(cls.PATHS["network_pkl"])
        elif cls.PATHS.get("adjacency"):
            graph = _load_npy_pair(cls.PATHS["positions_npy"], cls.PATHS["adjacency"])
        else:
            from graphs import read_graph_csv

            edge_list, positions = cls._network_paths(dataset_id)
            graph = read_graph_csv(edge_list, positions)

        # Whatever the format, the turn is applied once, here: a preview and
        # the run that follows it have to be looking at the same network.
        orient_positions(graph, cls.INPUT_ORIENTATION)
        return graph

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
    def load_original_image(cls, dataset_id: DatasetId) -> Optional[ndarray]:
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

        from utils import resize_image

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Image could not be read: {path}")
            return None

        # Measured, not declared: IMAGE_SIZE is the file's true size, in
        # (height, width) as it has always been written.
        cls.IMAGE_SIZE = (image.shape[0], image.shape[1])
        # Scaled to the frame, the way every CLI config does on load, so the
        # frame really is the size the background is drawn at.  A frame left
        # at the image's own size makes this a no-op.
        return resize_image(image, cls.FRAME_SIZE)
