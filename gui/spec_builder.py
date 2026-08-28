from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from configs.base_config import BaseConfig
from configs.gui_config import (
    NETWORK_FORMATS,
    PLOT_FORMATS,
    SPEC_CONTRACT_VERSION,
    format_param,
)

_FORMAT_HELP = {
    "csv": "…_edgelist.csv + …_positions.csv — what every loader here reads.",
    "pkl": "The graph itself, pickled. Reloadable as a single network.",
    "nkbin": "Compact binary plus …_positions.npy. Worth it for a big network.",
    "webp": "Lossless and small.",
    "png": "Lossless, larger, opens anywhere.",
    "svg": "Vector, for a figure that has to scale.",
}


#: The label each group of format switches shares.  Named here because the
#: window draws these rows on a plate of their own, apart from the other
#: output rows, and has to know which they are.
_FORMAT_ROWS = {"network": "Network format", "plot": "Plot format"}
FORMAT_ROWS = tuple(_FORMAT_ROWS.values())


def _format_fields(group: str, formats, default: str, omit: tuple = ()) -> List[Field]:
    noun = _FORMAT_ROWS[group]
    return [
        Field(
            format_param(group, name),
            f".{name}",
            name == default,
            kind="bool",
            group=OUTPUT_GROUP,
            row=noun,
            help=_FORMAT_HELP[name],
        )
        for name in formats
        if name not in omit
    ]


OUTPUT_GROUP = "Output"

SECTION_NOTES = {
    "Quality": (
        "Generation draws each network at random, so candidates differ in how "
        "closely they resemble the original. A candidate is measured against "
        "the input and generated again until it is within tolerance, or until "
        "the attempts run out — the closest one is kept either way."
    ),
    "Tiling": (
        "A large network is assembled rather than grown in one piece: seed "
        "points are scattered over the output area, a patch is generated "
        "around each of them, and growth then carries on from the patches' "
        "edges until the gaps between them close. The area is measured in "
        "backgrounds: a target scale of 100 × 100 is 100 of them down by 100 "
        "across."
    ),
    "Tracking": (
        "wandb needs a login: run `wandb login` once, or set WANDB_API_KEY in "
        "the environment. Switched off, nothing is contacted — the sweep "
        "writes sweep_report.csv either way."
    ),
}

ALIGN_GROUP = "Align"


@dataclass(frozen=True)
class Field:

    id: str
    label: str
    value: Any
    kind: str = "number"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    help: str = ""
    group: str = "Network"
    row: str = ""
    options: tuple = ()
    labels: tuple = ()
    option_help: tuple = ()
    hide_when: tuple = ()
    unit: str = ""
    axes: tuple = ("w", "h")

    def option_names(self) -> tuple:
        return self.labels or tuple(str(option) for option in self.options)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Tuples cross into QML as an opaque wrapper with no length and no
        # indexOf, which makes a combo box bound to one an invalid model.
        for key in ("options", "labels", "option_help", "axes"):
            data[key] = list(getattr(self, key))
        data.pop("hide_when")
        return data


def lines_of(fields: List[Field]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    shared: Dict[str, Dict[str, Any]] = {}
    for spec_field in fields:
        if not spec_field.row:
            lines.append({"row": "", "fields": [spec_field.as_dict()]})
            continue
        if spec_field.row not in shared:
            shared[spec_field.row] = {"row": spec_field.row, "fields": []}
            lines.append(shared[spec_field.row])
        shared[spec_field.row]["fields"].append(spec_field.as_dict())
    return lines


def sections_of(fields: List[Field]) -> List[Dict[str, Any]]:
    order: List[str] = []
    grouped: Dict[str, List[Field]] = {}
    for spec_field in fields:
        if spec_field.group not in grouped:
            order.append(spec_field.group)
            grouped[spec_field.group] = []
        grouped[spec_field.group].append(spec_field)
    return [
        {
            "name": name,
            "note": SECTION_NOTES.get(name, ""),
            "lines": lines_of(grouped[name]),
        }
        for name in order
    ]


@dataclass(frozen=True)
class Input:

    id: str
    label: str
    kind: str = "file"
    filter: str = ""
    placeholder: str = ""
    optional: bool = False
    help: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


SINGLE, DIRECTORY = "single", "directory"

SCOPE_LABELS = {SINGLE: "Single network", DIRECTORY: "Directory"}


@dataclass(frozen=True)
class InputShape:

    label: str
    inputs: List[Input]
    scope: str = SINGLE
    format: str = "CSV pair"

    @property
    def ids(self) -> tuple:
        return tuple(
            spec_input.id for spec_input in self.inputs if not spec_input.optional
        )

    @property
    def all_ids(self) -> tuple:
        return tuple(spec_input.id for spec_input in self.inputs)


@dataclass
class ModeSpec:

    name: str
    label: str
    blurb: str = ""
    fields: List[Field] = field(default_factory=list)
    input_shapes: List[InputShape] = field(default_factory=list)


_GATES = {
    "multifractal": (
        "Multifractal spectrum",
        "Compares the whole network's multifractal spectrum against the "
        "original's. The most searching of the three, and by far the slowest: "
        "every candidate costs an all-pairs shortest-path pass.",
    ),
    "length_angle": (
        "Edge length and angle",
        "Compares mean edge length and mean branching angle. Fast, but local — "
        "it says nothing about connectivity, so a candidate can pass here and "
        "still be wired quite unlike the original.",
    ),
    "none": (
        "No checking",
        "Every candidate is kept as it was generated. Nothing is measured, so "
        "a run takes as little time as it can.",
    ),
}


def _registered_checkers() -> tuple:
    from analysis.error_checker import _CHECKERS

    described = [name for name in _GATES if name in _CHECKERS]
    return tuple(described + sorted(set(_CHECKERS) - set(described)))


_ORIENTATIONS_NAMED = {
    "none": "Leave as it is",
    "rot90": "Rotate 90° clockwise",
    "rot180": "Rotate 180°",
    "rot270": "Rotate 90° anticlockwise",
    "transpose": "Transpose (swap x and y)",
}


def _common_fields(vector_plots: bool = True) -> List[Field]:
    from utils.graph_ops import ORIENTATIONS

    _ORIENTATION_LABELS = tuple(
        _ORIENTATIONS_NAMED.get(name, name) for name in ORIENTATIONS
    )

    return (
        [
            Field(
                "INPUT_ORIENTATION",
                "Turn network",
                "none",
                kind="choice",
                options=ORIENTATIONS,
                labels=_ORIENTATION_LABELS,
                group=ALIGN_GROUP,
                help=(
                    "For position data recorded in a different orientation to\n"
                    "the image it was traced from. Turned inside its own\n"
                    "bounding box, so only the extents swap.\n"
                    "The image is never turned: scale it with the input frame,\n"
                    "or prepare it beforehand."
                ),
            ),
            Field(
                "FRAME_SIZE",
                "Background image",
                (512, 512),
                kind="size",
                unit="px",
                help=(
                    "The coordinate window everything shares: the input image\n"
                    "is scaled to it, and the network is drawn in it.\n"
                    "Defaults to the image's own size, which is the reliable\n"
                    "answer — position data need not reach the corners, so its\n"
                    "extent understates the window."
                ),
            ),
            Field(
                "SYNTHETIC_FRAME_SIZE",
                "Synthetic frame",
                (512, 512),
                kind="size",
                unit="px",
                help=(
                    "Area a generated network grows into. Matching the input\n"
                    "frame is what makes the two comparable."
                ),
            ),
            Field(
                "CLOSED_NODES_FACTOR",
                "Node factor",
                1.2,
                minimum=0.1,
                maximum=5.0,
                step=0.1,
                help="Node-merge distance, relative to mean edge length.",
            ),
            Field(
                "CLOSED_EDGES_FACTOR",
                "Edge factor",
                0.8,
                minimum=0.1,
                maximum=5.0,
                step=0.1,
                help="Edge-crossing distance, relative to mean edge length.",
            ),
            Field(
                "SYNTHETIC_NETWORK_NUMBER",
                "Number of networks",
                1,
                kind="integer",
                minimum=1,
                maximum=500,
                step=1,
                help="How many synthetic networks this run produces.",
            ),
            Field(
                "SEED",
                "Seed",
                0,
                kind="integer",
                minimum=0,
                maximum=2**31 - 1,
                step=1,
                help="Same seed reproduces the same networks. 0 means unseeded.",
            ),
            Field(
                "ERROR_CHECKER",
                "Measured by",
                "multifractal",
                kind="choice",
                options=_CHECKER_NAMES,
                labels=_CHECKER_LABELS,
                option_help=_CHECKER_HELP,
                group="Quality",
                help=(
                    "How a candidate network is compared with the original "
                    "before it is kept."
                ),
            ),
            Field(
                "ERROR_TOLERANCE",
                "Tolerance",
                0.15,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                group="Quality",
                hide_when=(("ERROR_CHECKER", ("none",)),),
                help="How far from the original a candidate may be and still pass.",
            ),
            Field(
                "MAX_ATTEMPTS",
                "Max attempts",
                10,
                kind="integer",
                minimum=1,
                maximum=100,
                step=1,
                group="Quality",
                hide_when=(("ERROR_CHECKER", ("none",)),),
                help="Tries per network before giving up on it.",
            ),
            Field(
                "MEASURE_WEIGHTED",
                "Edge weights",
                False,
                kind="bool",
                group="Quality",
                hide_when=(("ERROR_CHECKER", ("none", "length_angle")),),
                help=(
                    "Measure along weighted paths rather than counting edges. "
                    "The input has to carry weights, or the run stops."
                ),
            ),
            Field(
                "FULL_Q_BAND",
                "Moment range",
                False,
                kind="choice",
                options=(False, True),
                labels=("Narrow", "Wide"),
                option_help=(
                    "Moments q from -3 to 3. Enough to tell candidates apart, "
                    "and the usual choice.",
                    "Moments q from -20 to 20. Reaches further into the sparse "
                    "and dense extremes of the network, and costs much more "
                    "for it.",
                ),
                group="Quality",
                hide_when=(("ERROR_CHECKER", ("none", "length_angle")),),
                help=(
                    "The span of moments the spectrum is measured over. Wider "
                    "weighs the extremes more heavily."
                ),
            ),
            Field(
                "SYNTHETIC_GRAPH_NUMBER",
                "Plot images",
                1,
                kind="integer",
                minimum=0,
                maximum=50,
                step=1,
                group=OUTPUT_GROUP,
                help="How many of the generated networks get a saved plot.",
            ),
            Field(
                "WRITE_SNAPSHOTS",
                "Snapshots",
                False,
                kind="bool",
                group=OUTPUT_GROUP,
                row="Snapshots",
                help=(
                    "Save the network as it grows, into a 'snapshots' folder\n"
                    "beside the results. Generate makes one network when this\n"
                    "is on, rather than the batch above."
                ),
            ),
            Field(
                "SNAPSHOT_INTERVAL",
                "Snapshot every",
                10,
                kind="integer",
                minimum=1,
                maximum=BaseConfig.PHASE2_MAX_ROUNDS,
                step=1,
                unit="rounds apart",
                group=OUTPUT_GROUP,
                row="Snapshots",
                hide_when=(("WRITE_SNAPSHOTS", (False,)),),
                help=(
                    "A round is one frontier of growth: every node waiting at\n"
                    "the start of it. The same in both modes — Generate grows\n"
                    "one network, Hybrid stitches between the patches."
                ),
            ),
        ]
        + _format_fields("network", NETWORK_FORMATS, "csv")
        + _format_fields(
            "plot", PLOT_FORMATS, "webp", omit=() if vector_plots else ("svg",)
        )
    )


def _hybrid_fields() -> List[Field]:
    return [
        Field(
            "TARGET_SCALE",
            "Target scale",
            (100, 100),
            kind="size",
            axes=("rows", "cols"),
            help=(
                "A multiplier, not a size. The assembled area is\n"
                "rows × background height by cols × background width —\n"
                "so 100 × 100 over a 510 px background is 51000 px square."
            ),
            group="Tiling",
        ),
        Field(
            "MIN_CENTER_DISTANCE_FACTOR",
            "Seed spacing",
            1.5,
            minimum=0.5,
            maximum=10.0,
            step=0.1,
            unit="× the background",
            help=(
                "How far apart the seeds are scattered, as a multiple of the\n"
                "background frame. It sets the scale of everything below it:\n"
                "the patch size is a fraction of this gap, and whatever is\n"
                "left of the gap is what stitching has to close."
            ),
            group="Tiling",
        ),
        Field(
            "NUM_CENTERS",
            "Patches",
            2000,
            kind="integer",
            minimum=1,
            maximum=100000,
            step=100,
            unit="at most",
            help=(
                "A ceiling on how many patches are placed. The spacing above\n"
                "decides how many actually fit, and scattering stops at\n"
                "whichever comes first — so a low ceiling leaves the far end\n"
                "of a large area empty, whatever the spacing says."
            ),
            group="Tiling",
        ),
        Field(
            "TILE_FRAME_FACTOR",
            "Patch size",
            0.5,
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            unit="× the gap between seeds",
            help=(
                "Each patch grows inside a square of its own, sized from the\n"
                "gap between its seed and the nearest other seed. At 0.5 a\n"
                "patch takes half that gap and stitching closes the rest;\n"
                "larger patches meet sooner and leave less to stitch.\n"
                "There is no size in pixels to set: no two seeds are closer\n"
                "than the spacing above, so the same fraction of that gap is\n"
                "the smallest a patch can be."
            ),
            group="Tiling",
        ),
    ]


_CHECKER_NAMES = _registered_checkers()
_CHECKER_LABELS = tuple(_GATES.get(name, (name, ""))[0] for name in _CHECKER_NAMES)
_CHECKER_HELP = tuple(_GATES.get(name, (name, ""))[1] for name in _CHECKER_NAMES)


def _background() -> Input:
    return Input(
        "image",
        "Image",
        filter="Images (*.tif *.tiff *.png *.jpg *.jpeg *.webp)",
        placeholder="optional background",
        optional=True,
        help=(
            "Any image OpenCV can read; used in greyscale.\n"
            "Drawn at its own pixel size inside the frame, so\n"
            "its width and height should match the range the\n"
            "positions cover — an image of another size lands\n"
            "beside the network rather than under it."
        ),
    )


def _network_shapes() -> List[InputShape]:
    return [
        InputShape(
            "One network (CSV pair)",
            format="CSV pair",
            inputs=[
                Input(
                    "edge_list",
                    "Edge list",
                    filter="CSV files (*.csv)",
                    placeholder="…_edgelist.csv",
                    help=(
                        "CSV, one edge per row.\n"
                        "Columns: source_index, target_index, and\n"
                        "edge_weight for a weighted network.\n"
                        "The indices are row numbers in the positions\n"
                        "file; one that names a node with no position\n"
                        "stops the run."
                    ),
                ),
                Input(
                    "positions",
                    "Positions",
                    filter="CSV files (*.csv)",
                    placeholder="…_positions.csv",
                    help=(
                        "CSV, one node per row.\n"
                        "Columns: x, y.\n"
                        "A row's number is the node index the edge\n"
                        "list refers to."
                    ),
                ),
                _background(),
            ],
        ),
        InputShape(
            "A directory of networks",
            scope=DIRECTORY,
            format="CSV pair",
            inputs=[
                Input(
                    "datasets_dir",
                    "Networks dir",
                    kind="dir",
                    placeholder="folder of …_edgelist.csv pairs",
                    help=(
                        "One dataset per …_edgelist.csv, each needing a\n"
                        "matching …_positions.csv beside it.\n"
                        "…_image.tif is used as the background when it\n"
                        "is there, and skipped when it is not.\n"
                        "An edge list with no positions stops the run\n"
                        "rather than being passed over."
                    ),
                )
            ],
        ),
        InputShape(
            "One network (NumPy pair)",
            format="NumPy pair",
            inputs=[
                Input(
                    "adjacency",
                    "Adjacency",
                    filter="NumPy files (*.npy)",
                    placeholder="…_mat.npy",
                    help=(
                        "…_mat.npy holding a scipy sparse adjacency\n"
                        "matrix, saved as a 0-d object array and read\n"
                        "with np.load(...).item().\n"
                        "A plain dense array is refused rather than\n"
                        "read as something else."
                    ),
                ),
                Input(
                    "positions_npy",
                    "Positions",
                    filter="NumPy files (*.npy)",
                    placeholder="…_pos.npy",
                    help=(
                        "…_pos.npy, shape (N, 2), one row per node.\n"
                        "Recorded as (row, column) and transposed to\n"
                        "(x, y) when loaded — the same thing every CLI\n"
                        "config does with a pair like this."
                    ),
                ),
                _background(),
            ],
        ),
        InputShape(
            "One network (pickle)",
            format="Pickle",
            inputs=[
                Input(
                    "network_pkl",
                    "Network",
                    filter="Pickle files (*.pkl)",
                    placeholder="…_network.pkl",
                    help=(
                        "A pickled SynthGraph, or a list of them — the\n"
                        "synthetic_network_*.pkl a run writes.\n"
                        "From a list the first is read, and the log\n"
                        "says so.\n"
                        "A legacy networkx pickle needs networkx\n"
                        "installed, which it is not by default."
                    ),
                ),
                _background(),
            ],
        ),
    ]


def _sweep_fields() -> List[Field]:
    swept = {
        "CLOSED_NODES_FACTOR",
        "CLOSED_EDGES_FACTOR",
        "SYNTHETIC_GRAPH_NUMBER",
        "WRITE_SNAPSHOTS",
        "SNAPSHOT_INTERVAL",
    }
    fields = []
    for spec_field in _common_fields():
        if spec_field.id in swept:
            continue
        if spec_field.id == "SYNTHETIC_NETWORK_NUMBER":
            spec_field = replace(
                spec_field,
                label="Networks per trial",
                value=1,
                help="Each factor combination generates this many networks.",
            )
        fields.append(spec_field)
    return fields + [
        Field(
            "USE_WANDB",
            "Log to wandb",
            True,
            kind="bool",
            group="Tracking",
            help=(
                "On, the sweep is run by wandb and needs a login: run\n"
                "`wandb login` once, or put WANDB_API_KEY (or WANDB_KEY)\n"
                "in the environment before starting the window.\n"
                "Off, the grid is walked here and nothing is contacted.\n"
                "Either way the run writes sweep_report.csv."
            ),
        ),
        Field(
            "NF_RANGE",
            "Node factor range",
            (1.0, 1.5),
            kind="range",
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            help=(
                "The first and last node factor tried.\n"
                "Every combination of node and edge value is one trial, so\n"
                "the count is node steps × edge steps."
            ),
        ),
        Field(
            "EF_RANGE",
            "Edge factor range",
            (1.0, 1.5),
            kind="range",
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            help="The first and last edge factor tried.",
        ),
        Field(
            "SWEEP_STEP",
            "Step",
            0.1,
            minimum=0.01,
            maximum=1.0,
            step=0.01,
            unit="per step",
            help=(
                "How far apart the values tried are, on both ranges.\n"
                "A smaller step means a finer grid and many more trials:\n"
                "1.0 → 1.5 is 6 values at 0.1, and 51 at 0.01."
            ),
        ),
    ]


NETWORK_INPUTS = frozenset(key for shape in _network_shapes() for key in shape.ids)


MODES: Dict[str, ModeSpec] = {
    "generate": ModeSpec(
        "generate",
        "Generate",
        "Build new networks with the statistics of one you already have.",
        _common_fields(),
        _network_shapes(),
    ),
    "hybrid": ModeSpec(
        "hybrid",
        "Hybrid",
        "Assemble a large network by tiling generated patches together.",
        # No .svg: a hybrid network is drawn as pixels by OpenCV, both for the
        # assembled image and for its snapshots, so there is no vector to write
        # and asking for one only fails the run at the end of it.
        _hybrid_fields() + _common_fields(vector_plots=False),
        _network_shapes(),
    ),
    "sweep": ModeSpec(
        "sweep",
        "Sweep",
        "Try a range of node and edge factors, scoring every combination.",
        _sweep_fields(),
        _network_shapes(),
    ),
}


def shape_for(mode: str, inputs: Dict[str, str]) -> InputShape:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    keys = set(inputs)
    for shape in MODES[mode].input_shapes:
        if set(shape.ids) <= keys <= set(shape.all_ids):
            return shape
    raise ValueError(
        f"{mode}: {sorted(keys)} matches none of its input shapes: "
        f"{[list(s.all_ids) for s in MODES[mode].input_shapes]}"
    )


_SAMPLES = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples")

_SAMPLE_INPUTS = {
    "edge_list": os.path.join(_SAMPLES, "gui_mode", "sample_1_edgelist.csv"),
    "positions": os.path.join(_SAMPLES, "gui_mode", "sample_1_positions.csv"),
    "image": os.path.join(_SAMPLES, "gui_mode", "sample_1_image.tif"),
    "datasets_dir": os.path.join(_SAMPLES, "gui_mode"),
    "adjacency": os.path.join(_SAMPLES, "generate_mode", "sample_1_mat.npy"),
    "positions_npy": os.path.join(_SAMPLES, "generate_mode", "sample_1_pos.npy"),
    "network_pkl": os.path.join(_SAMPLES, "gui_mode", "sample_1_network.pkl"),
}


PROJECT_ROOT = BaseConfig.PROJECT_ROOT


def display_path(path: str) -> str:
    if not path:
        return ""
    try:
        inside = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path
    return path if inside.startswith(os.pardir) else inside


def resolve_path(text: str) -> str:
    if not text:
        return ""
    return os.path.abspath(os.path.join(PROJECT_ROOT, os.path.expanduser(text)))


def sample_for(key: str) -> str:
    path = _SAMPLE_INPUTS.get(key, "")
    return path if path and os.path.exists(path) else ""


def input_state(spec_input: Input, path: str) -> Tuple[str, str]:
    if not path:
        if spec_input.optional:
            return "blank", ""
        return "missing", f"Choose the {spec_input.label.lower()}."
    if not os.path.exists(path):
        return "missing", f"{spec_input.label} not found: {path}"
    if spec_input.kind == "dir" and not os.path.isdir(path):
        return "missing", f"{spec_input.label} is not a directory: {path}"
    return "ok", ""


def default_values(mode: str) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    return {f.id: f.value for f in MODES[mode].fields}


def default_inputs(mode: str, shape_index: int = 0) -> Dict[str, str]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    shape = MODES[mode].input_shapes[shape_index]
    return {spec_input.id: sample_for(spec_input.id) for spec_input in shape.inputs}


def validate(
    mode: str, inputs: Dict[str, str], output_dir: str, values: Dict[str, Any]
) -> List[str]:
    problems: List[str] = []

    if mode not in MODES:
        problems.append(f"Unknown mode: {mode}")
        return problems

    for spec_input in shape_for(mode, inputs).inputs:
        _, problem = input_state(spec_input, inputs.get(spec_input.id) or "")
        if problem:
            problems.append(problem)

    if not output_dir:
        problems.append("Choose an output directory.")

    for spec_field in MODES[mode].fields:
        if spec_field.id not in values:
            continue
        value = values[spec_field.id]
        bounded = value if isinstance(value, (list, tuple)) else [value]
        for one in bounded:
            if not isinstance(one, (int, float)):
                continue
            if spec_field.minimum is not None and one < spec_field.minimum:
                problems.append(f"{spec_field.label}: below {spec_field.minimum}")
            if spec_field.maximum is not None and one > spec_field.maximum:
                problems.append(f"{spec_field.label}: above {spec_field.maximum}")

    for key in ("NF_RANGE", "EF_RANGE"):
        span = values.get(key)
        if isinstance(span, (list, tuple)) and span[0] > span[1]:
            problems.append(f"{key.replace('_', ' ').title()}: start is above end.")

    plot_formats = [
        format_param("plot", name)
        for name in PLOT_FORMATS
        if format_param("plot", name) in values
    ]
    wanted_images = values.get("SYNTHETIC_GRAPH_NUMBER") or 0
    if plot_formats and wanted_images and not any(values[k] for k in plot_formats):
        problems.append(
            f"Plot images is {wanted_images:g} but no plot format is chosen, so "
            "none would be written. Pick a format, or set plot images to 0."
        )

    for spec_field in MODES[mode].fields:
        if spec_field.kind != "choice" or spec_field.id not in values:
            continue
        if values[spec_field.id] not in spec_field.options:
            problems.append(
                f"{spec_field.label} must be one of: "
                f"{', '.join(spec_field.option_names())}."
            )

    return problems


def build_spec(
    mode: str,
    inputs: Dict[str, str],
    output_dir: str,
    values: Dict[str, Any],
    image: Optional[str] = None,
    run_name: str = "gui_run",
) -> Dict[str, Any]:
    params = dict(values)

    if params.get("SEED") in (0, None):
        params["SEED"] = None

    frame = params.get("SYNTHETIC_FRAME_SIZE")
    if frame is not None:
        params.setdefault("FRAME_SIZE", frame)

    if "NF_RANGE" in params:
        params.setdefault("CLOSED_NODES_FACTOR", params["NF_RANGE"][0])
        params.setdefault("CLOSED_EDGES_FACTOR", params["EF_RANGE"][0])

    params = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }

    shape = shape_for(mode, inputs)
    spec_inputs = {
        spec_input.id: (
            os.path.abspath(inputs[spec_input.id])
            if inputs.get(spec_input.id)
            else None
        )
        for spec_input in shape.inputs
    }
    if "image" in shape.all_ids:
        chosen = image or inputs.get("image")
        spec_inputs["image"] = os.path.abspath(chosen) if chosen else None

    return {
        "contract": SPEC_CONTRACT_VERSION,
        "mode": mode,
        "run_name": run_name,
        "output_dir": os.path.abspath(output_dir),
        "inputs": spec_inputs,
        "params": params,
    }


def write_spec(spec: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(spec, handle, indent=2)
    return path
