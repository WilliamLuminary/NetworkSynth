# NetworkSynth — code release

Analysis and synthesis of spatial networks. This directory is the minimal, self-contained code release accompanying the manuscript: it holds the two pipelines behind the reported results, and nothing else.

Licensed under Apache-2.0 (`LICENSE`). Input datasets are **not** included — see [Input data](#input-data).

## Installation

Python 3.10 or newer (tested on 3.12), Linux or macOS. All dependencies are open source and installed from PyPI; no proprietary software is required.

```bash
sudo apt-get install -y cmake libgomp1   # for networkit; see its docs for macOS
pip install -r requirements.txt
```

Then, once input data is in place:

```bash
python -m run generate
python -m run hybrid
```

`hybrid` is memory-hungry at scale — a large run can need 100 GB of RAM and its second phase takes on the order of minutes to hours.

## The two pipelines

```bash
python -m run generate    # synthetic networks matching each original
python -m run hybrid      # one large network: seed tiles + frontier continuation
```

**`generate`** loads each original network, saves its structural properties and figures, then synthesizes networks in parallel. Every candidate must pass a multifractal quality gate against the original before it is accepted; failures are retried up to `MAX_ATTEMPTS` times.

**`hybrid`** builds one large network in two phases. Phase 1 scatters seed centers across the canvas and grows an independent network *tile* around each, in parallel. Phase 2 assembles the tiles on a shared canvas and continues growth from each tile's frontier nodes to fill the seams between them.

Which pipeline to run is the **only** command-line argument. Everything else is set on the config class, so a run is fully described by its config file plus `RANDOM_SEED`.

## Input data

**Input datasets are not distributed with this code release.** Download the data, point the configuration at it, then run.

Each dataset is a set of files named after a dataset ID:

| File | Contents |
| --- | --- |
| `<id>_pos.npy` | node positions, float array of shape `(N, 2)` |
| `<id>_mat.npy` | adjacency matrix, shape `(N, N)` |
| `<id>_image.tif` | background micrograph (optional; may be absent) |

The shipped configs expect them laid out per pipeline:

```
data/input/samples/
├── generate_mode/   sample_1, sample_2, sample_3            (run generate)
└── hybrid_mode/     sample_A, sample_B, sample_C, sample_D  (run hybrid)
```

Data elsewhere does not need moving — set `BASE_INPUT_PATH` on the config instead. Dataset IDs are filename stems, so `DatasetId("sample_1")` resolves to `sample_1_pos.npy` and `sample_1_mat.npy` inside `BASE_INPUT_PATH`.

**CSV input.** Networks stored as a node-position table plus an edge list are also supported. Both sample configs carry the CSV loaders as a commented-out block below the `.npy` ones — uncomment them, add `import csv` at the top of the file, and adjust `load_original_network()` to pass the node count as the comment shows. Expected files:

| File | Contents |
| --- | --- |
| `<id>_NodePositions.csv` | one header row, then `x,y` per node; row order = node ID |
| `<id>_EdgeList.csv` | one header row, then `Source,Target,Weight`; each undirected edge once, the loader symmetrizes |

For any other layout, replace the config's `load_original_network()` and `load_original_image()` hooks — see [Configuration](#configuration).

## Random seed

Generation is stochastic: the growth process draws node degrees, branch angles, and edge lengths at random. `RANDOM_SEED` in `configs/base_config.py` (default 42) sets the seed; `None` draws from OS entropy instead.

Within a batch, each network or tile is seeded from its own index, so a batch of *N* networks is a set of *N* independent draws rather than *N* copies of one.

## Configuration

Each pipeline reads one config:

| Pipeline | Config |
| --- | --- |
| `generate` | `configs/generate_mode/config_sample.py` |
| `hybrid` | `configs/hybrid_mode/config_sample.py` |

`configs/base_config.py` holds `BaseConfig`, which every config inherits and which defines the shared defaults (`RANDOM_SEED`, `MAX_ATTEMPTS`, `ERROR_CHECKER`, worker counts, output paths).

To run on your own data, edit the config: point `BASE_INPUT_PATH` at your files, list your dataset IDs in `DATASETS`, and adjust the parameters below. The two loader hooks are what tie a config to data on disk:

```python
class SampleConfig(BaseConfig):
    DATASETS = [DatasetId("sample_1"), DatasetId("sample_2")]
    BASE_INPUT_PATH = "/path/to/your/data"

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        """Return a SynthGraph, e.g. via build_graph(positions, adjacency)."""
        ...

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        """Return the background image as a numpy array, or None."""
        ...
```

`DatasetId` identifies a dataset and maps to its filename stem or directory:

```python
from configs import DatasetId

DatasetId("sample_1").path     # "sample_1"       single-level
DatasetId("A", "10kX").path    # "A/10kX"         multi-level (set + resolution)
```

### Key parameters

| Parameter | Typical values | Description |
| --- | --- | --- |
| `RANDOM_SEED` | any int, or `None` | Seed for reproducible runs. `None` = unseeded. |
| `CLOSED_NODES_FACTOR` | 0.5–2.0 | Node merging likelihood |
| `CLOSED_EDGES_FACTOR` | 0.5–2.0 | Edge proximity tolerance |
| `ERROR_CHECKER` | `multifractal` / `none` | Quality-gate algorithm; `none` skips checking |
| `ERROR_TOLERANCE` | 0.1–0.5 | Similarity threshold (used by `multifractal`) |
| `SYNTHETIC_NETWORK_NUMBER` | 1–100 | Networks to synthesize per dataset (`generate`) |
| `SYNTHETIC_GRAPH_NUMBER` | 0–10 | How many of them to render (≤ network count) |
| `MAX_ATTEMPTS` | 5–20 | Retry attempts per network |
| `SNAPSHOT_INTERVAL` | 0, or N | 0 = off; N = save a growth snapshot every N new nodes |

### Hybrid: tile frame sizing

Each Phase 1 tile grows inside a **frame** — a square boundary telling the tile when to stop and hand off to Phase 2. The frame size therefore controls how wide a seam Phase 2 has to stitch: a smaller frame leaves a wider seam, a larger one makes tiles nearly touch. Two modes:

**Fixed** — set `TILE_FRAME_SIZE = (w, h)` and every tile uses that frame. Deterministic, no measurement.

**Auto** (default, `TILE_FRAME_SIZE = None`) — each tile is sized from the distance `d` to its nearest neighboring center:

```
frame side = max(d * TILE_FRAME_FACTOR, MIN_TILE_FRAME)
```

Tiles in crowded regions get smaller frames, tiles in open space larger ones — each scaled to the room it actually has.

| Parameter | Default | Description |
| --- | --- | --- |
| `TILE_FRAME_SIZE` | `None` | Fixed frame `(w, h)`. `None` enables auto mode. |
| `TILE_FRAME_FACTOR` | `0.5` | Auto: frame side as a fraction of the nearest-neighbor distance. Lower = wider seam for Phase 2 to stitch. |
| `MIN_TILE_FRAME` | `382.0` | Auto: floor on the frame side, so closely-spaced centers still produce tiles large enough to clear `MIN_TILE_NODES`. |

The Phase 1 log (tag `PHASE1`) prints the resulting frame-side min/max, so you can confirm tiles are not shrunk below a usable size.

## What is in this release

```
run.py            Entry point; the only argument is which pipeline to run
pipelines/        generate.py and hybrid.py
graphs/           SynthGraph (NetworKit-backed), the BFS growth generator,
                  the node model
configs/          BaseConfig + the generate, hybrid, and analysis configs
handlers/         Data loading, structural attribute calculation, edge-weight
                  mapping, run orchestration, output serialization
analysis/         Multifractal analyzer and the error checker used as the
                  generation quality gate
utils.py          Graph construction, metrics, rendering, snapshots
```

Development-only material — other pipeline modes, notebooks, helper scripts, and the test suites — stays in the development repository.

## Graph representation: SynthGraph

`graphs/synth_graph.py` defines the primary graph type. It pairs a NetworKit graph (C++ engine, holding structure and edge weights) with a NumPy positions array indexed by integer node ID:

```
SynthGraph
  ├── nk.Graph          # structure + edge weights
  └── np.ndarray (N,2)  # node positions
```

Key methods: `positions()`, `degree()`, `neighbors()`, `edges()`, `weight()`, `set_weight()`, `largest_connected_component()`, `subgraph()`, `copy()`, `to_networkx()`, `from_networkx()`, `from_sparse_matrix()`, `from_graph_nodes()`.

All graph algorithms (Dijkstra, betweenness, closeness, eigenvector centrality, diameter, connected components, clustering) use NetworKit. NetworkX is a dependency only for reading legacy `nx.Graph` pickles — `from_networkx()` in `graphs/synth_graph.py` and the backward-compatible loader in `configs/analyze_mode/config_sample.py`.

## Output and the save system

Saving is config-driven: each config declares **what** formats to write, and `handlers/saver.py` handles **how** (path construction, timestamps, directory creation).

```
Pipeline  →  Saver.save(content, identifier, prefix)
                 ↓
              Config.save(identifier)  →  returns list of specs
                 ↓
              Serializer(content, filepath)  →  writes to disk
```

**Serializers** (`configs/file_definitions.py`) are pure `(content, filepath)` functions — the available output formats:

| Serializer | Output | Accepts |
| --- | --- | --- |
| `save_pickle` | `.pkl` | any Python object |
| `save_csv` | `.csv` | rows (list of lists) |
| `save_text` | `.txt` | a string |
| `save_webp` | `.webp` | matplotlib Figure, ndarray, or PIL image |
| `save_png` | `.png` | matplotlib Figure, ndarray, or PIL image |
| `save_svg` | `.svg` (vector) | matplotlib Figure only |
| `save_network_csv` | `_edgelist.csv` (+ weights) + `_positions.csv` | a SynthGraph |
| `save_network_nkbin` | `.nkbin` + `_positions.npy` | a SynthGraph |
| `save_networkit` | `.nkbin` + companion `_positions.npy` | `nk.Graph` or `(graph, positions)` |

Rendered output comes in exactly two kinds: a matplotlib `Figure` (the only kind that can also be written as vector `.svg`) or a BGR `ndarray` (raster).

**Specs.** Each `save_<identifier>()` classmethod returns a list of tuples, one tuple per file:

```python
(relative_dir, detail, extension, save_fn)   # optional 5th element: False = no timestamp
```

The Saver builds the path as `{output_dir}/{relative_dir}/{prefix}{detail}_{timestamp}.{extension}`.

`DEFAULT_SAVE_SPECS` in `configs/file_definitions.py` gives every standard identifier a default, inherited by all configs. Default policy: images → `.webp`, network exports → `.csv`.

| Identifier | Directory | Format | Serializer |
| --- | --- | --- | --- |
| `original_image` | `original/` | `.webp` | `save_webp` |
| `original_network` | `original/` | `.csv` | `save_network_csv` |
| `original_property` | `original/` | `.pkl` | `save_pickle` |
| `original_graph` | `original/` | `.webp` | `save_webp` |
| `original_report` | `original/` | `.txt` (no timestamp) | `save_text` |
| `synthetic_graph` | `synthetic/` | `.webp` | `save_webp` |
| `synthetic_network` | `synthetic/` | `.pkl` | `save_pickle` |
| `synthetic_export` | `synthetic/` | `.csv` | `save_network_csv` |
| `synthetic_report` | `synthetic/` | `.txt` (no timestamp) | `save_text` |
| `analysis_data` | root | `.pkl` (no timestamp) | `save_pickle` |
| `analysis_figure` | root | `.webp` | `save_webp` |

`save_svg` and `save_network_nkbin` are available but not on by default.

**To change an output format, touch only the config.** Define `save_<identifier>()` on your config class and return the specs you want; the dispatcher (`BaseConfig.save`) prefers it over `DEFAULT_SAVE_SPECS`. Only override the identifiers you want to change.

```python
class SampleConfig(BaseConfig):

    # Vector SVG instead of the default WebP:
    @classmethod
    def save_original_graph(cls):
        from ..file_definitions import save_svg
        return [("original", "original_graph", "svg", save_svg)]

    # Both CSV and the binary .nkbin (+ .npy):
    @classmethod
    def save_synthetic_export(cls):
        from ..file_definitions import save_network_csv, save_network_nkbin
        return [
            ("synthetic", "synthetic_network", "csv", save_network_csv),
            ("synthetic", "synthetic_network", "nkbin", save_network_nkbin),
        ]
```

To add a new output type, add a spec to `DEFAULT_SAVE_SPECS` (or define `save_<identifier>()` on the config) and call it from the pipeline:

```python
data_agent.saver.save(my_data, "my_custom_data")
```

Identifiers are plain strings; there is no enum to register.

## Logs

Every run emits structured JSON lines to `data/output/logs/`:

```json
{
  "ts": "2026-03-27 22:24:37",
  "run_id": "304868d0",
  "level": "INFO",
  "logger": "handlers.saver",
  "message": "Create new directory: ...",
  "tag": "IO",
  "dataset": "sample_A"
}
```

| Field | Description |
| --- | --- |
| `ts` | Timestamp |
| `run_id` | 8-char hex ID unique per run (safe for concurrent appends) |
| `level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `logger` | Module that emitted the record |
| `message` | Human-readable message |
| `tag` | *optional* — `PHASE1`, `PHASE2`, `SNAPSHOT`, `IO`, `MEMORY`, `CONFIG`, `PIPELINE` |
| `dataset` | *optional* — dataset identifier |

Useful `jq` queries:

```bash
LOG=data/output/logs/project_SampleConfig.jsonl

jq 'select(.run_id == "304868d0")' "$LOG"       # one run
jq 'select(.tag == "PHASE1")' "$LOG"            # one phase
jq 'select(.level == "ERROR") | {message, exception}' "$LOG"
jq -s 'group_by(.tag) | map({tag: .[0].tag, count: length})' "$LOG"
tail -f "$LOG" | jq --unbuffered 'select(.tag == "PHASE1")'   # follow a live run
```

## Troubleshooting

**Missing input files** — check that the paths match the config's `BASE_INPUT_PATH` and its dataset IDs.

**Dimension mismatch** — node positions must be `(N, 2)` and the adjacency matrix `(N, N)`, with the same `N`.

**Generation failures** — loosen the constraints in the config:

```python
CLOSED_NODES_FACTOR *= 1.2  # allow more node merging
ERROR_TOLERANCE *= 1.5      # accept less similar networks
MAX_ATTEMPTS = 20           # more retries per network
```
