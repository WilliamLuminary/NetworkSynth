from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

from configs.gui_config import SPEC_CONTRACT_VERSION


@dataclass(frozen=True)
class Field:

    id: str
    label: str
    value: Any
    kind: str = "number"  # number | integer | bool | text | size
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    help: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Input:

    id: str
    label: str
    kind: str = "file"  # file | dir
    filter: str = ""
    placeholder: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputShape:
    """One way a mode can be given its input."""

    label: str
    inputs: List[Input]

    @property
    def ids(self) -> tuple:
        return tuple(spec_input.id for spec_input in self.inputs)


@dataclass
class ModeSpec:

    name: str
    label: str
    fields: List[Field] = field(default_factory=list)
    #: The ways this mode can be given its input, in order.  Kept in step with
    #: ``MODE_INPUTS``, which is what the run-spec is validated against.
    input_shapes: List[InputShape] = field(default_factory=list)


def _registered_checkers() -> tuple:
    from analysis.error_checker import _CHECKERS

    return tuple(_CHECKERS)


def _common_fields() -> List[Field]:
    return [
        Field(
            "SYNTHETIC_FRAME_SIZE",
            "Synthetic frame (w × h)",
            (512, 512),
            kind="size",
            help="Area the generated network grows into.",
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
            "Networks to generate",
            5,
            kind="integer",
            minimum=1,
            maximum=500,
            step=1,
        ),
        Field(
            "SYNTHETIC_GRAPH_NUMBER",
            "Preview images",
            1,
            kind="integer",
            minimum=0,
            maximum=50,
            step=1,
        ),
        Field(
            "MAX_ATTEMPTS",
            "Max attempts per network",
            10,
            kind="integer",
            minimum=1,
            maximum=100,
            step=1,
        ),
        Field(
            "ERROR_TOLERANCE",
            "Error tolerance",
            0.15,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            help="Multifractal quality gate; ignored when the gate is off.",
        ),
        Field(
            "ERROR_CHECKER",
            "Quality gate",
            "multifractal",
            kind="text",
            help='"multifractal", "length_angle" or "none". Off is much faster.',
        ),
        Field("MEASURE_WEIGHTED", "Weighted analysis", False, kind="bool"),
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
    ]


def _hybrid_fields() -> List[Field]:
    return [
        Field(
            "TARGET_SCALE",
            "Target scale (w × h)",
            (100, 100),
            kind="size",
            help="How many tile-widths of network to assemble.",
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
        ),
        Field(
            "TILE_FRAME_FACTOR",
            "Tile frame factor",
            0.5,
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            help="Tile frame side = nearest-neighbour distance x this.",
        ),
        Field(
            "MIN_TILE_FRAME",
            "Minimum tile frame",
            382.0,
            minimum=1.0,
            maximum=5000.0,
            step=10.0,
        ),
    ]


#: Gate names ``create_error_checker`` accepts.  Derived, so a new checker does
#: not have to be remembered here as well.
_CHECKER_NAMES = tuple(sorted(_registered_checkers()))


def _network_shapes() -> List[InputShape]:
    return [
        InputShape(
            "One network",
            [
                Input(
                    "edge_list",
                    "Edge list",
                    filter="CSV files (*.csv)",
                    placeholder="…_edgelist.csv",
                ),
                Input(
                    "positions",
                    "Positions",
                    filter="CSV files (*.csv)",
                    placeholder="…_positions.csv",
                ),
            ],
        ),
        InputShape(
            "A directory of networks",
            [
                Input(
                    "datasets_dir",
                    "Networks dir",
                    kind="dir",
                    placeholder="every …_edgelist.csv + …_positions.csv pair in it",
                )
            ],
        ),
    ]


def _results_shapes() -> List[InputShape]:
    return [
        InputShape(
            "A results directory",
            [
                Input(
                    "networks_dir",
                    "Results dir",
                    kind="dir",
                    placeholder="a directory holding synthetic/ and original/",
                )
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
            help="Use edge weights as distances. Needs a weighted network.",
        ),
        Field(
            "FULL_Q_BAND",
            "Full q band",
            False,
            kind="bool",
            help="Wider q range: slower, smoother spectrum.",
        ),
    ]


#: Only modes the entry point can actually dispatch.  Kept in step with
#: ``gui_run._MODES`` — a mode offered here that cannot run is worse than one
#: that is simply absent.
#:
#: Each mode gets the common fields plus whatever its pipeline reads that
#: ``BaseConfig`` does not define.  Those extras are not optional: the pipeline
#: reads them as plain attributes, so a missing one is an ``AttributeError``
#: partway through a run rather than a rejected spec.
MODES: Dict[str, ModeSpec] = {
    "generate": ModeSpec("generate", "Generate", _common_fields(), _network_shapes()),
    "hybrid": ModeSpec(
        "hybrid", "Hybrid", _common_fields() + _hybrid_fields(), _network_shapes()
    ),
    "sweep": ModeSpec("sweep", "Sweep", _sweep_fields(), _network_shapes()),
    "analyze": ModeSpec("analyze", "Analyze", _analysis_fields(), _results_shapes()),
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
        if set(shape.ids) == keys:
            return shape
    raise ValueError(
        f"{mode}: {sorted(keys)} matches none of its input shapes: "
        f"{[list(s.ids) for s in MODES[mode].input_shapes]}"
    )


def default_values(mode: str) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    return {f.id: f.value for f in MODES[mode].fields}


def default_inputs(mode: str, shape_index: int = 0) -> Dict[str, str]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    shape = MODES[mode].input_shapes[shape_index]
    return {spec_input.id: "" for spec_input in shape.inputs}


def validate(
    mode: str, inputs: Dict[str, str], output_dir: str, values: Dict[str, Any]
) -> List[str]:
    problems: List[str] = []

    if mode not in MODES:
        problems.append(f"Unknown mode: {mode}")
        return problems

    for spec_input in shape_for(mode, inputs).inputs:
        path = inputs.get(spec_input.id) or ""
        if not path:
            problems.append(f"Choose the {spec_input.label.lower()}.")
        elif not os.path.exists(path):
            problems.append(f"{spec_input.label} not found: {path}")
        elif spec_input.kind == "dir" and not os.path.isdir(path):
            problems.append(f"{spec_input.label} is not a directory: {path}")

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

    checker = values.get("ERROR_CHECKER")
    if checker is not None and checker not in _CHECKER_NAMES:
        problems.append(f"Quality gate must be one of: {', '.join(_CHECKER_NAMES)}.")

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
    # caller says otherwise; IMAGE_SIZE only matters when there is an image.
    frame = params.get("SYNTHETIC_FRAME_SIZE")
    if frame is not None:
        params.setdefault("FRAME_SIZE", frame)
        params.setdefault("IMAGE_SIZE", frame)

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
    if "edge_list" in shape.ids:
        spec_inputs["image"] = os.path.abspath(image) if image else None

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
