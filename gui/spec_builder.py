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

#: What each output format is for.  Keyed by the format tables above, so a
#: format added there without a word about it fails loudly here.
_FORMAT_HELP = {
    "csv": "…_edgelist.csv + …_positions.csv — what every loader here reads.",
    "pkl": "The graph itself, pickled. Reloadable as a single network.",
    "nkbin": "Compact binary plus …_positions.npy. Worth it for a big network.",
    "webp": "Lossless and small.",
    "png": "Lossless, larger, opens anywhere.",
    "svg": "Vector, for a figure that has to scale.",
}


def _format_fields(group: str, formats, default: str) -> List[Field]:
    """One switch per format this group can be written in.

    Derived from the tables the config writes by, so the form offers exactly
    what the save layer can do — no more, and nothing it has forgotten.
    """
    noun = {"network": "Networks", "plot": "Plots"}[group]
    return [
        Field(
            format_param(group, name),
            f".{name}",
            name == default,
            kind="bool",
            group=OUTPUT_GROUP,
            # One line of switches per kind of output, rather than a line each.
            row=noun,
            help=_FORMAT_HELP[name],
        )
        for name in formats
    ]


#: The section holding everything about what a run writes.  Named here rather
#: than in the window, because the window shows it in its own pane and has to
#: know which one it is.
OUTPUT_GROUP = "Output"

#: The section for squaring the input up with the image it was traced from.
#: Its own pane, beside the files it is about.
ALIGN_GROUP = "Align"


@dataclass(frozen=True)
class Field:

    id: str
    label: str
    value: Any
    kind: str = "number"  # number | integer | bool | text | choice | size | range
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    help: str = ""
    #: Which section of the form this belongs under.
    group: str = "Network"
    #: Fields sharing one render side by side under this label, as a set of
    #: switches rather than a switch per line.
    row: str = ""
    #: For ``kind="choice"``: everything the field accepts.
    options: tuple = ()
    #: The one number in its section that a run is usually about, drawn larger.
    prominent: bool = False
    #: What the number is counted in, shown after the editor.  Not every size
    #: is a length: a target scale counts tiles.
    unit: str = ""
    #: What the two halves of a ``size`` are, in order.  Not every pair is a
    #: width and a height — hybrid reads its target scale as (rows, columns).
    axes: tuple = ("w", "h")

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # A tuple crosses into QML as an opaque wrapper rather than an array:
        # it has no length and no indexOf, so a combo box bound to one is an
        # invalid model, the same way a tuple value has to be listed too.
        data["options"] = list(self.options)
        data["axes"] = list(self.axes)
        return data


def lines_of(fields: List[Field]) -> List[Dict[str, Any]]:
    """*fields* as the rows a form draws: one per field, or one per ``row``."""
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
    """*fields* grouped into sections, in the order they first appear."""
    order: List[str] = []
    grouped: Dict[str, List[Field]] = {}
    for spec_field in fields:
        if spec_field.group not in grouped:
            order.append(spec_field.group)
            grouped[spec_field.group] = []
        grouped[spec_field.group].append(spec_field)
    return [{"name": name, "lines": lines_of(grouped[name])} for name in order]


@dataclass(frozen=True)
class Input:

    id: str
    label: str
    kind: str = "file"  # file | dir
    filter: str = ""
    placeholder: str = ""
    #: A run goes ahead without it.  Left out of ``InputShape.ids``, so an
    #: optional input never becomes something the run-spec demands.
    optional: bool = False
    #: What this file has to contain, shown behind the row's info button.
    #: Wrapped by hand: a tooltip does not wrap for itself.
    help: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: The two things an input can be, kept apart from what format it is in: one
#: network to work on, or a folder of them to work through.
SINGLE, DIRECTORY = "single", "directory"

SCOPE_LABELS = {SINGLE: "Single network", DIRECTORY: "Directory"}


@dataclass(frozen=True)
class InputShape:
    """One way a mode can be given its input."""

    label: str
    inputs: List[Input]
    #: One network or a folder of them.  Editing is offered only for one.
    scope: str = SINGLE
    #: What the files are, said separately from how many there are.
    format: str = "CSV pair"

    @property
    def ids(self) -> tuple:
        """What this shape cannot run without — the keys ``MODE_INPUTS`` names."""
        return tuple(
            spec_input.id for spec_input in self.inputs if not spec_input.optional
        )

    @property
    def all_ids(self) -> tuple:
        """Every key the form offers, optional ones included."""
        return tuple(spec_input.id for spec_input in self.inputs)


@dataclass
class ModeSpec:

    name: str
    label: str
    #: One line under the title saying what this mode does.
    blurb: str = ""
    fields: List[Field] = field(default_factory=list)
    #: The ways this mode can be given its input, in order.  Kept in step with
    #: ``MODE_INPUTS``, which is what the run-spec is validated against.
    input_shapes: List[InputShape] = field(default_factory=list)


def _registered_checkers() -> tuple:
    from analysis.error_checker import _CHECKERS

    return tuple(_CHECKERS)


def _common_fields() -> List[Field]:
    from utils.graph_ops import ORIENTATIONS

    return (
        [
            Field(
                "INPUT_ORIENTATION",
                "Turn network",
                "none",
                kind="choice",
                options=ORIENTATIONS,
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
            ),
            Field(
                "SYNTHETIC_NETWORK_NUMBER",
                "Number of networks",
                5,
                kind="integer",
                minimum=1,
                maximum=500,
                step=1,
                prominent=True,
                help="How many synthetic networks this run produces.",
            ),
            Field(
                "ERROR_CHECKER",
                "Gate",
                "multifractal",
                kind="choice",
                options=_CHECKER_NAMES,
                group="Quality",
                help="What a candidate is checked against. Off is much faster.",
            ),
            Field(
                "ERROR_TOLERANCE",
                "Tolerance",
                0.15,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                group="Quality",
                help="How far from the original a candidate may be.",
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
                help="Tries per network before giving up on it.",
            ),
            Field(
                "MEASURE_WEIGHTED",
                "Weighted analysis",
                False,
                kind="bool",
                group="Quality",
            ),
            Field(
                "SEED",
                "Seed",
                0,
                kind="integer",
                minimum=0,
                maximum=2**31 - 1,
                step=1,
                group="Quality",
                help="Same seed reproduces the same networks. 0 means unseeded.",
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
        ]
        + _format_fields("network", NETWORK_FORMATS, "csv")
        + _format_fields("plot", PLOT_FORMATS, "webp")
    )


def _hybrid_fields() -> List[Field]:
    return [
        Field(
            "TARGET_SCALE",
            "Target scale",
            (100, 100),
            kind="size",
            unit="× the background",
            # `scale_rows, scale_cols = config.TARGET_SCALE` — down first,
            # across second, which is the opposite way round to every frame.
            axes=("rows", "cols"),
            help=(
                "A multiplier, not a size. The assembled area is\n"
                "rows × background height by cols × background width —\n"
                "so 100 × 100 over a 510 px background is 51000 px square."
            ),
            group="Tiling",
        ),
        Field(
            "PHASE2_MAX_ROUNDS",
            "Phase 2 max rounds",
            500,
            kind="integer",
            minimum=1,
            maximum=10000,
            step=50,
            help="Frontier continuation after the seed tiles are placed.",
            group="Tiling",
        ),
        Field(
            "TILE_FRAME_FACTOR",
            "Tile frame factor",
            0.5,
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            help="Tile frame side = nearest-neighbour distance x this.",
            group="Tiling",
        ),
        Field(
            "MIN_TILE_FRAME",
            "Minimum tile frame",
            382.0,
            unit="px",
            minimum=1.0,
            maximum=5000.0,
            step=10.0,
            group="Tiling",
        ),
    ]


#: Gate names ``create_error_checker`` accepts.  Derived, so a new checker does
#: not have to be remembered here as well.
_CHECKER_NAMES = tuple(sorted(_registered_checkers()))


def _background() -> Input:
    """The image a network was traced from, drawn behind it.

    Optional everywhere: a network is plottable and analysable without one.  A
    directory of networks gets this for free instead, by the `…_image.tif`
    convention, so only the single-network shapes offer it.
    """
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
    """Every way one of these modes can be handed the network to copy.

    One entry per format the loaders read, in the order ``MODE_INPUTS`` lists
    them.  A format the app cannot load is worse offered than absent, so
    ``.nkbin`` is not here: it is written, and read back only by
    ``scripts/helpers/load_network_example.py``.
    """
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
            # The only format a folder can be read as: discover_datasets scans
            # for `*_edgelist.csv` and its positions partner, nothing else.
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


def _results_shapes() -> List[InputShape]:
    return [
        InputShape(
            "Two sets to compare",
            scope=DIRECTORY,
            format="Result folders",
            inputs=[
                Input(
                    "original_dir",
                    "Original",
                    kind="dir",
                    placeholder="original/ folder",
                    help=(
                        "A folder of networks, read as a batch .pkl if\n"
                        "one is in it, and otherwise as every\n"
                        "…_edgelist.csv + …_positions.csv pair.\n"
                        "A run's own original/ folder is what this\n"
                        "expects."
                    ),
                ),
                Input(
                    "synthetic_dir",
                    "Synthetic",
                    kind="dir",
                    placeholder="synthetic/ folder",
                    help=(
                        "The same, for the networks being compared\n"
                        "against the originals — a run's synthetic/\n"
                        "folder."
                    ),
                ),
            ],
        )
    ]


def _sweep_fields() -> List[Field]:
    # The factors are what a sweep varies, so they come from the ranges below
    # rather than from a fixed field, and no preview images are written.
    swept = {"CLOSED_NODES_FACTOR", "CLOSED_EDGES_FACTOR", "SYNTHETIC_GRAPH_NUMBER"}
    fields = []
    for spec_field in _common_fields():
        if spec_field.id in swept:
            continue
        if spec_field.id == "SYNTHETIC_NETWORK_NUMBER":
            spec_field = replace(
                spec_field,
                label="Networks per trial",
                value=20,
                help="Each factor combination generates this many networks.",
            )
        fields.append(spec_field)
    return fields + [
        Field(
            "NF_RANGE",
            "Node factor range",
            (1.0, 1.5),
            kind="range",
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            help="Swept in steps of 0.1. Trials = node steps x edge steps.",
        ),
        Field(
            "EF_RANGE",
            "Edge factor range",
            (1.0, 1.5),
            kind="range",
            minimum=0.1,
            maximum=5.0,
            step=0.1,
        ),
    ]


def _analysis_fields() -> List[Field]:
    # Analysis is multifractal only, so there is no quality gate to choose.
    return [
        Field(
            "MEASURE_WEIGHTED",
            "Weighted analysis",
            False,
            kind="bool",
            group="Analysis",
            help="Use edge weights as distances. Needs a weighted network.",
        ),
        Field(
            "FULL_Q_BAND",
            "Full q band",
            False,
            kind="bool",
            group="Analysis",
            help="Wider q range: slower, smoother spectrum.",
        ),
        # No network formats: comparison writes spectra, not networks.
    ] + _format_fields("plot", PLOT_FORMATS, "webp")


#: Only modes the entry point can actually dispatch.  Kept in step with
#: ``gui_run._GUI_MODES`` — a mode offered here that cannot run is worse than one
#: that is simply absent.
#:
#: Each mode gets the common fields plus whatever its pipeline reads that
#: ``BaseConfig`` does not define.  Those extras are not optional: the pipeline
#: reads them as plain attributes, so a missing one is an ``AttributeError``
#: partway through a run rather than a rejected spec.
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
        _common_fields() + _hybrid_fields(),
        _network_shapes(),
    ),
    "sweep": ModeSpec(
        "sweep",
        "Sweep",
        "Try a range of node and edge factors, scoring every combination.",
        _sweep_fields(),
        _network_shapes(),
    ),
    "compare": ModeSpec(
        "compare",
        "Compare",
        "Measure two sets of finished networks against each other.",
        _analysis_fields(),
        _results_shapes(),
    ),
}


def shape_for(mode: str, inputs: Dict[str, str]) -> InputShape:
    """The shape *inputs* belongs to, identified by its keys.

    The keys carry the choice, so nothing has to pass the shape alongside the
    values and risk the two disagreeing.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    keys = set(inputs)
    for shape in MODES[mode].input_shapes:
        # Every required key, and nothing the shape does not offer: an optional
        # key may be absent, but a key from the *other* shape must not decide
        # this one.
        if set(shape.ids) <= keys <= set(shape.all_ids):
            return shape
    raise ValueError(
        f"{mode}: {sorted(keys)} matches none of its input shapes: "
        f"{[list(s.all_ids) for s in MODES[mode].input_shapes]}"
    )


_SAMPLES = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples")

#: What each input starts as.  Only used when the sample is actually there: a
#: checkout has these, a release built without ``data/input`` does not, and an
#: empty box beats one pointing at a file nobody shipped.
_SAMPLE_INPUTS = {
    "edge_list": os.path.join(_SAMPLES, "gui_mode", "sample_1_edgelist.csv"),
    "positions": os.path.join(_SAMPLES, "gui_mode", "sample_1_positions.csv"),
    "image": os.path.join(_SAMPLES, "gui_mode", "sample_1_image.tif"),
    "datasets_dir": os.path.join(_SAMPLES, "gui_mode"),
    "adjacency": os.path.join(_SAMPLES, "generate_mode", "sample_1_mat.npy"),
    "positions_npy": os.path.join(_SAMPLES, "generate_mode", "sample_1_pos.npy"),
}


PROJECT_ROOT = BaseConfig.PROJECT_ROOT


def display_path(path: str) -> str:
    """*path* as the form shows it: relative to the project when it is inside.

    Only how it is written, never what is stored: a spec always carries the
    absolute path, because it is read by a subprocess and again by whatever
    that spawns, and those do not share a working directory.

    A relative path already means "relative to the project" here — the GUI
    starts every child with the project root as its working directory — so
    this is the shorter half of the same name, not a second convention.
    """
    if not path:
        return ""
    try:
        inside = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        # Windows, different drive: there is no relative form.
        return path
    return path if inside.startswith(os.pardir) else inside


def resolve_path(text: str) -> str:
    """What the form was given, as the absolute path everything else uses."""
    if not text:
        return ""
    return os.path.abspath(os.path.join(PROJECT_ROOT, os.path.expanduser(text)))


def sample_for(key: str) -> str:
    path = _SAMPLE_INPUTS.get(key, "")
    return path if path and os.path.exists(path) else ""


def input_state(spec_input: Input, path: str) -> Tuple[str, str]:
    """How one input stands, and what to say when it is wrong.

    The single reading of it: the form marks the row from this and the run is
    refused by this, so a row cannot show a tick while the run says otherwise.
    """
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

    # No format at all is a choice, not a mistake: a run made only to look at
    # the result wants nothing on disk.  Asking for plot images in no format is
    # a different thing — that request cannot be met, so say so.
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

    # Every choice field, not the quality gate alone: each one declares what
    # it accepts, so nothing has to be listed twice here.
    for spec_field in MODES[mode].fields:
        if spec_field.kind != "choice" or spec_field.id not in values:
            continue
        if values[spec_field.id] not in spec_field.options:
            problems.append(
                f"{spec_field.label} must be one of: "
                f"{', '.join(spec_field.options)}."
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

    # A seed of 0 means "do not seed": SEED=None is how the pipelines say that,
    # but a spin box cannot hold None.
    if params.get("SEED") in (0, None):
        params["SEED"] = None

    # The frame the original is measured in matches the synthetic one unless a
    # caller says otherwise.
    frame = params.get("SYNTHETIC_FRAME_SIZE")
    if frame is not None:
        params.setdefault("FRAME_SIZE", frame)

    # IMAGE_SIZE is not derived here any more: it is the input image's true
    # size, which only the loader can know, and it measures it as it reads.

    # Every trial replaces these with the combination being scored, but
    # SynthParams still has to be built from the config before that happens, so
    # the sweep's own form does not offer them.
    if "NF_RANGE" in params:
        params.setdefault("CLOSED_NODES_FACTOR", params["NF_RANGE"][0])
        params.setdefault("CLOSED_EDGES_FACTOR", params["EF_RANGE"][0])

    # Every size field, not a named few: modes add their own (TILE_FRAME_SIZE,
    # TARGET_SCALE), and a list is what survives the JSON round trip.
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
        # Either source, one key: *image* for a caller that has a path in hand,
        # the form's own optional input for the GUI.
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
