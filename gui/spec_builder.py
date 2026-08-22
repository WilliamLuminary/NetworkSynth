from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
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


@dataclass
class ModeSpec:

    name: str
    label: str
    fields: List[Field] = field(default_factory=list)


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
            help='"multifractal" or "none". Turning it off is much faster.',
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


#: Only modes the entry point can actually dispatch.  Kept in step with
#: ``gui_run._MODES`` — a mode offered here that cannot run is worse than one
#: that is simply absent.
#:
#: Each mode gets the common fields plus whatever its pipeline reads that
#: ``BaseConfig`` does not define.  Those extras are not optional: the pipeline
#: reads them as plain attributes, so a missing one is an ``AttributeError``
#: partway through a run rather than a rejected spec.
MODES: Dict[str, ModeSpec] = {
    "generate": ModeSpec("generate", "Generate", _common_fields()),
    "hybrid": ModeSpec("hybrid", "Hybrid", _common_fields() + _hybrid_fields()),
}


def default_values(mode: str) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; available: {sorted(MODES)}")
    return {f.id: f.value for f in MODES[mode].fields}


def validate(
    mode: str, edge_list: str, positions: str, output_dir: str, values: Dict[str, Any]
) -> List[str]:
    problems: List[str] = []

    if mode not in MODES:
        problems.append(f"Unknown mode: {mode}")
        return problems

    if not edge_list:
        problems.append("Choose an edge-list CSV.")
    elif not os.path.exists(edge_list):
        problems.append(f"Edge list not found: {edge_list}")

    if not positions:
        problems.append("Choose a positions CSV.")
    elif not os.path.exists(positions):
        problems.append(f"Positions file not found: {positions}")

    if not output_dir:
        problems.append("Choose an output directory.")

    for spec_field in MODES[mode].fields:
        if spec_field.id not in values:
            continue
        value = values[spec_field.id]
        if spec_field.minimum is not None and isinstance(value, (int, float)):
            if value < spec_field.minimum:
                problems.append(f"{spec_field.label}: below {spec_field.minimum}")
        if spec_field.maximum is not None and isinstance(value, (int, float)):
            if value > spec_field.maximum:
                problems.append(f"{spec_field.label}: above {spec_field.maximum}")

    checker = values.get("ERROR_CHECKER")
    if checker is not None and checker not in ("multifractal", "none"):
        problems.append('Quality gate must be "multifractal" or "none".')

    return problems


def build_spec(
    mode: str,
    edge_list: str,
    positions: str,
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

    # Every size field, not a named few: modes add their own (TILE_FRAME_SIZE,
    # TARGET_SCALE), and a list is what survives the JSON round trip.
    params = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }

    return {
        "contract": SPEC_CONTRACT_VERSION,
        "mode": mode,
        "run_name": run_name,
        "output_dir": os.path.abspath(output_dir),
        "inputs": {
            "edge_list": os.path.abspath(edge_list) if edge_list else None,
            "positions": os.path.abspath(positions) if positions else None,
            "image": os.path.abspath(image) if image else None,
        },
        "params": params,
    }


def write_spec(spec: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(spec, handle, indent=2)
    return path
