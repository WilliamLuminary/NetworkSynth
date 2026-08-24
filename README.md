# Graph Network Analysis and Synthesis Toolkit

Network Analysis
Synthetic Generation

## Quick Start

1. **Install and Run**
  ```bash
   git clone https://github.com/WilliamLuminary/NetworkSynth.git
   cd NetworkSynth
   pip install -r requirements.txt  # Python 3.11-3.13

   python run.py configs/generate_mode/config_sample.py
  ```

   Python 3.14 is not supported yet: NetworKit ships no cp314 wheel, and neither do the numpy, scipy and matplotlib versions pinned here, so a 3.14 install has to compile from source. See `INTEGRATION_PLAN.md`, Appendix A.

2. **Prepare Sample Data**
  ```
   data/input/samples/generate_mode/
   ├── sample_1_pos.npy    # Node positions (N, 2)
   ├── sample_1_mat.npy    # Adjacency matrix
   └── sample_1_image.tif  # Background image (optional)
  ```
3. **Expected Output**

   One directory per run, named `{MODE}_{ConfigClass}_results_{timestamp}_{run_id}`, with `data/output/latest_result` repointed at it:

  ```
   data/output/generate_mode_SampleConfig_results_20260824_130049_c6a6a6ec/
   ├── manifest.json                       # what the run produced, and how it ended
   ├── run.jsonl                           # this run's structured log
   └── sample_1/
       ├── original/
       │   ├── original_graph_*.webp
       │   ├── original_image_*.webp
       │   ├── original_network_*_edgelist.csv
       │   ├── original_network_*_positions.csv
       │   ├── original_property_*.pkl
       │   └── report.txt
       └── synthetic/
           ├── synthetic_graph_*.webp
           ├── synthetic_network_*.pkl
           ├── synthetic_network_*_edgelist.csv
           ├── synthetic_network_*_positions.csv
           └── report.txt
  ```

   `manifest.json` is written from a `finally`, so it exists even for a run that
   failed or was interrupted — its `status` is `ok`, `failed` or `cancelled`, and
   `outputs` groups every file the run wrote by kind.

   A run with `DISABLE_SAVING` set creates none of this, not even the directory.

## Configuration System

### DatasetId

The unified `DatasetId` class replaces the old `SetName`/`Resolution` enum system:

```python
from configs import DatasetId

# Single-level dataset (e.g., nanowires)
dataset = DatasetId("1-0")
dataset.path    # "1-0"
dataset[0]      # "1-0"

# Multi-level dataset (e.g., set + resolution)
dataset = DatasetId("A", "10kX")
dataset.path    # "A/10kX" (OS-appropriate)
dataset[0]      # "A"
dataset[1]      # "10kX"
```

### Creating a New Config

1. Create `configs/generate_mode/config_mydata.py`:

```python
from typing import List
from ..base_config import BaseConfig
from ..dataset_id import DatasetId

def _generate_datasets() -> List[DatasetId]:
    return [DatasetId("set_1"), DatasetId("set_2")]

class ConfigMydata(BaseConfig):
    MODE = "generate"          # which pipeline runs this config
    DATASETS = _generate_datasets()

    IMAGE_SIZE = (1024, 1024)
    FRAME_SIZE = (512, 512)
    SYNTHETIC_FRAME_SIZE = (1536, 1536)

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    SYNTHETIC_NETWORK_NUMBER = 10
    SYNTHETIC_GRAPH_NUMBER = 3
    ERROR_TOLERANCE = 0.15

    # Required, no default: measure edge widths, or only topology.  Asking for
    # widths a network does not carry raises rather than quietly downgrading.
    MEASURE_WEIGHTED = True

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        # Load positions + adjacency, return SynthGraph
        # via build_graph(positions, matrix)
        ...

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        # Load and return image as numpy array (or None)
        ...
```

2. Set `MODE` on it so a pipeline claims it (`generate`, `hybrid`, `mosaic`,
   `scaling`, `sweep`, or `compare`), then run it:

```bash
python run.py configs/generate_mode/config_mydata.py
```

The file is the name — nothing has to be registered or exported.

### Save System

Saving is config-driven: each config declares **what** formats to save, and the `Saver` handles **how** (path construction, timestamps, directory creation).

#### Three layers

```
Pipeline  →  Saver.save(content, identifier, prefix)
                 ↓
              Config.save(identifier)  →  a tuple of SaveSpec
                 ↓
              spec.save_fn(content, filepath)  →  writes to disk
```

**Layer 1 — Serializers** (`configs/file_definitions.py`): Pure `(content, filepath)` functions. These are the *available formats* — pick them in a config spec (Layer 2).

| Serializer | Output | Accepts |
| --- | --- | --- |
| `save_pickle` | `.pkl` | any Python object |
| `save_csv` | `.csv` | rows (list of lists) |
| `save_text` | `.txt` | a string |
| `save_webp` | `.webp` | matplotlib Figure **or** ndarray **or** PIL image |
| `save_png` | `.png` | matplotlib Figure **or** ndarray **or** PIL image |
| `save_svg` | `.svg` (vector) | matplotlib Figure only |
| `save_network_csv` | `_edgelist.csv` + `_positions.csv` | a SynthGraph |
| `save_network_nkbin` | `.nkbin` + `_positions.npy` | a SynthGraph |
| `save_networkit` | `.nkbin` + companion `_positions.npy` | `nk.Graph` or `(graph, positions)` |

Rendered visual output comes in exactly two canonical kinds: a **matplotlib `Figure`** (the only kind that can also be saved as vector `.svg`) or a **BGR `ndarray`** (raster). `save_webp`/`save_png` accept either.

`save_network_csv` writes the `edge_weight` column **only for a weighted graph**.
An unweighted NetworKit graph reports every weight as `1.0`, so writing the
column unconditionally produced a file that read back as weighted — and both the
`Mapper` (which decides whether to assign widths at all) and the analyzer
(weighted vs topological measure) act on that answer. The reader treats the
column's presence as the answer, so weightedness now survives a round trip
truthfully in both directions.

**Layer 2 — Save specs** (`BaseConfig.SAVE_<IDENTIFIER>`): one attribute per
output, holding a tuple of `SaveSpec`. Every setting is named, so a spec reads as
what it is:

```python
SaveSpec(ORIGINAL_DIR, "report", "txt", save_text, use_timestamp=False)
```

`SaveSpec` is a frozen dataclass in `configs/file_definitions.py`:

| Field | Meaning |
| --- | --- |
| `relative_dir` | Subdirectory under the run's output root (`""` = the root) |
| `detail` | Descriptive stem in the filename |
| `extension` | File extension, without the dot |
| `save_fn` | Serialiser, called as `save_fn(content, filepath)` |
| `use_timestamp` | Whether the filename carries the run's timestamp (default `True`) |

**Layer 3 — Saver** (`handlers/saver.py`): single public method `save()`. Gets
specs from the config, builds file paths, creates directories, and calls the
serialiser with `(content, filepath)`. The path is
`{output_dir}/{relative_dir}/{prefix}{detail}_{timestamp}.{extension}`.

#### The defaults

`BaseConfig` declares one `SAVE_*` attribute per output. Every mode config
inherits them. **Default policy: images → `.webp`, network exports → `.csv`.**

| Identifier | Attribute | Directory | Format |
| --- | --- | --- | --- |
| `original_image` | `SAVE_ORIGINAL_IMAGE` | `original/` | `.webp` |
| `original_graph` | `SAVE_ORIGINAL_GRAPH` | `original/` | `.webp` |
| `original_network` | `SAVE_ORIGINAL_NETWORK` | `original/` | `.csv` |
| `original_property` | `SAVE_ORIGINAL_PROPERTY` | `original/` | `.pkl` |
| `original_report` | `SAVE_ORIGINAL_REPORT` | `original/` | `.txt`, no timestamp |
| `synthetic_graph` | `SAVE_SYNTHETIC_GRAPH` | `synthetic/` | `.webp` |
| `synthetic_network` | `SAVE_SYNTHETIC_NETWORK` | `synthetic/` | `.pkl` |
| `synthetic_export` | `SAVE_SYNTHETIC_EXPORT` | `synthetic/` | `.csv` |
| `synthetic_report` | `SAVE_SYNTHETIC_REPORT` | `synthetic/` | `.txt`, no timestamp |
| `analysis_data` | `SAVE_ANALYSIS_DATA` | root | `.pkl`, no timestamp |
| `analysis_figure` | `SAVE_ANALYSIS_FIGURE` | root | `.webp` |

`save_svg` (vector) and `save_network_nkbin` (`.nkbin` + `.npy`) are **not**
defaults but remain available — name them in an override.

#### Changing what a config writes

**Only the config changes** — no pipeline or serialiser edits. Declare the
attribute; ordinary class inheritance does the rest, and everything you do not
mention stays on the default.

```python
from ..file_definitions import (
    ORIGINAL_DIR,
    SYNTHETIC_DIR,
    SaveSpec,
    save_network_csv,
    save_network_nkbin,
    save_png,
    save_svg,
    save_webp,
)


class ConfigDickson(BaseConfig):

    # Vector SVG instead of the default WebP:
    SAVE_ORIGINAL_GRAPH = (
        SaveSpec(ORIGINAL_DIR, "original_graph", "svg", save_svg),
    )

    # One save call, two files — csv and the binary .nkbin (+ .npy):
    SAVE_SYNTHETIC_EXPORT = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "csv", save_network_csv),
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "nkbin", save_network_nkbin),
    )

    # webp AND png:
    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "png", save_png),
    )
```

One `SaveSpec` = one file. List several to emit several formats for the same
output. Pick the format by picking the `save_fn` and its `extension` from the
serialiser table above.

#### Adding a brand-new output type

Identifiers are plain strings — nothing to register:

1. Declare `SAVE_MY_CUSTOM_DATA` on `BaseConfig` (or just on your config):

```python
SAVE_MY_CUSTOM_DATA = (SaveSpec("custom_dir", "my_data", "pkl", save_pickle),)
```

2. Call it from the pipeline with that identifier:

```python
run.save(my_data, "my_custom_data")
```

`BaseConfig.save("my_custom_data")` resolves `SAVE_MY_CUSTOM_DATA`. If none of
the existing serialisers fit, add a `(content, filepath)` function in
`configs/file_definitions.py` and name it in the spec.

### Render System

Saving decides *where a file goes*; rendering decides *how the picture looks*.
The two are deliberately separate, and rendering follows the same shape.

**The paradigm** — `RenderStyle` in `configs/file_definitions.py` names every
field you can set. It is a frozen, flat dataclass: any output may set any subset,
so "a plot with a background image" needs no special class.

| Field | Meaning |
| --- | --- |
| `node_size` | Marker diameter in **points** (matplotlib's `markersize` unit). `None` = thinnest the renderer can draw |
| `line_width` | Edge width in points. `None` = thinnest |
| `alpha` | Opacity of a background image, for outputs that have one |
| `dpi` | `None` lets the renderer choose from the node count |
| `max_px` | Cap on the longest side in pixels |
| `margin_frac` | Fractional margin around the bounding box |
| `border` | Draw a rectangle around the bounding box |
| `show_on_the_fly` | Preview on screen while the run is in progress |

The OpenCV renderers convert points to pixels at `dpi` (a point is 1/72 inch), so
one number means the same thing in every renderer. Note that a sub-point value
at a high dpi still rounds to one pixel.

**The defaults** — `BaseConfig` declares one `RENDER_*` attribute per rendered
output. These are the generate mode's: matplotlib plots at 300 dpi with 6pt
markers. The OpenCV renderers (hybrid, mosaic, scaling) leave `node_size` and
`line_width` unset, drawing single-pixel nodes and 1px edges — worth setting once
a network is large.

| Identifier | Attribute | Renderer |
| --- | --- | --- |
| `original_graph` | `RENDER_ORIGINAL_GRAPH` | matplotlib |
| `synthetic_graph` | `RENDER_SYNTHETIC_GRAPH` | matplotlib |
| `bfs_snapshot` | `RENDER_BFS_SNAPSHOT` | matplotlib |
| `hybrid_snapshot` | `RENDER_HYBRID_SNAPSHOT` | OpenCV |
| `hybrid_graph` | `RENDER_HYBRID_GRAPH` | OpenCV |
| `mosaic_graph` | `RENDER_MOSAIC_GRAPH` | OpenCV (`border=True`) |
| `scaled_graph` | `RENDER_SCALED_GRAPH` | OpenCV |

**Overriding** — declare the attribute, changing only the fields you care about:

```python
from dataclasses import replace


class ConfigDickson(BaseConfig):

    # Dense gel networks read better with smaller dots.
    RENDER_ORIGINAL_GRAPH = replace(BaseConfig.RENDER_ORIGINAL_GRAPH, node_size=3.0)

    # At the 16383px WebP limit an 11M-node network is ~190 megapixels, which
    # most viewers refuse to open.
    RENDER_HYBRID_GRAPH = replace(BaseConfig.RENDER_HYBRID_GRAPH, max_px=8000)
```

`RenderStyle` is **frozen**, so a config cannot mutate a shared style and
restyle every other config's plots in the process — `replace()` returns a new
one. Pipelines read a style with `config.render("hybrid_graph")`, the twin of
`config.save("synthetic_export")`.

### Choosing a config

A config is named by its file, and each file holds exactly one config class:

```bash
python run.py configs/generate_mode/config_snapshot_1x1.py
python run.py configs/hybrid_mode/config_snapshot.py
```

The config is the only argument: it carries its datasets, its parameters, and a
`MODE` naming the pipeline that runs it, so there is no mode argument to keep in
step with the file. `ls configs/*_mode/config_*.py` is the list of what you can
pass, and only the file you name is imported.


## Analysing Networks That Already Exist

Generating a network and measuring one are different jobs, so measuring has its
own entry point — `analyse.py` — and no config module at all. There is nothing
to configure but where to read, where to write, and how to measure:

```bash
python analyse.py                                  # uses the constants in the script
python analyse.py <input> [<input> ...] <out_dir>   # last argument is the output directory
```

An input is a directory of networks, a single `*_edgelist.csv`, or a `.pkl`
batch. Each input is analysed as **its own labelled set**, so several inputs
produce one figure per measure with every set drawn on it. Labels come from the
input paths, taking on parent directories as needed to stay distinct — two runs'
own `original` directories become `sample_A_original` and `sample_B_original`
rather than colliding.

Four constants at the top of the script hold the defaults; the two paths can be
overridden on the command line:

| Constant | Meaning |
| --- | --- |
| `INPUTS` | Default input paths |
| `OUTPUT_DIR` | Default output directory |
| `MEASURE_WEIGHTED` | Measure edge widths, or topology only |
| `FULL_Q_BAND` | Wide q band (401 points, `-20..20`) instead of the narrow one (61 points, `-3..3`) |

Output is a plain directory — no timestamped run root, no `latest_result`, no
manifest:

```
analysis_data.pkl        # {results: {label: [...]}, measure_weighted, full_q_band}
analysis_spectra.webp    # f(alpha) vs alpha, one curve per network
analysis_dimensions.webp # D(q) vs q
```

The measurement settings are stored **inside** the data file, because the same
networks measured weighted and topologically give different answers and nothing
else in the file distinguishes them.

`MEASURE_WEIGHTED` is required rather than inferred: a weighted network measured
as pure topology is a legitimate choice, so it cannot be read off the data.
Asking for widths a network does not carry raises immediately instead of
quietly measuring something else.


## Key Parameters

| Parameter                  | Typical Values | Description                           |
| -------------------------- | -------------- | ------------------------------------- |
| `CLOSED_NODES_FACTOR`      | 0.5-2.0        | Node merging likelihood               |
| `CLOSED_EDGES_FACTOR`      | 0.5-2.0        | Edge proximity tolerance              |
| `ERROR_CHECKER`            | `multifractal` / `none` | Quality-gate algorithm; `none` skips checking |
| `ERROR_TOLERANCE`          | 0.1-0.5        | Similarity threshold (used by `multifractal`) |
| `SYNTHETIC_NETWORK_NUMBER` | 1-100          | Networks to generate per dataset      |
| `SYNTHETIC_GRAPH_NUMBER`   | 0-10           | Graphs to visualize (≤ network count) |
| `MAX_ATTEMPTS`             | 5-20           | Retry attempts per network            |


## Hybrid Mode: Tile Frame Sizing

Hybrid mode (`configs/hybrid_mode/config_sample.py`) builds one large network in two phases:

- **Phase 1** scatters many *seed centers* across the canvas and grows an independent network *tile* around each one, in parallel.
- **Phase 2** stitches the tiles together, continuing growth from each tile's frontier nodes to fill the gaps between them.

Each tile grows inside a **frame** — a square boundary that tells the tile when to stop growing and hand off to Phase 2. The frame size therefore controls how much of the gap between neighboring tiles is left for Phase 2 to fill: a smaller frame leaves a wider seam to stitch, a larger frame makes tiles nearly touch.

There are two ways to set it, selected in `configs/hybrid_mode/config_sample.py`:

**Fixed** — set `TILE_FRAME_SIZE = (w, h)` and every tile uses that same frame. Deterministic; no measurement.

**Auto** (default, `TILE_FRAME_SIZE = None`) — each tile is sized from the distance `d` to its nearest neighboring center:

```
frame side = max(d * TILE_FRAME_FACTOR, MIN_TILE_FRAME)
```

Tiles in crowded regions get smaller frames and tiles in open space get larger ones — each scaled to the room it actually has.

| Parameter | Default | Description |
| --- | --- | --- |
| `TILE_FRAME_SIZE` | `None` | Fixed frame `(w, h)` for every tile. `None` enables auto mode. |
| `TILE_FRAME_FACTOR` | `0.5` | Auto mode: frame side as a fraction of the nearest-neighbor distance. Lower = wider seam for Phase 2 to stitch; higher = tiles nearly touch. |
| `MIN_TILE_FRAME` | `382.0` | Auto mode: floor on the frame side, so closely-spaced centers still produce tiles large enough to clear `MIN_TILE_NODES`. |

The Phase 1 log (tag `PHASE1`) prints the resulting frame-side min/max for each run, so you can confirm tiles aren't being shrunk below a usable size.


## Sweep Best Parameters

Best `(CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR)` per dataset from hyperparameter sweeps.

| Dataset | Node Factor | Edge Factor | Error | Success Rate | Note                         |
| ------- | ----------- | ----------- | ----- | ------------ | ---------------------------- |
| A       | 1.0         | 1.5         |       |              | (local test: yaxing_li/hyperparam-tuning/runs/n1tl0yfr)         |
| B       | 1.8         | 1.5         | 0.073 | 30%          | Verified in local deployment |
| B       | 0.7         | 2.0         | 0.054 | 36%          | Lowest error w/ usable success (best combined) |
| B       | 0.4         | 2.0         | 0.086 | 46%          | Balanced |
| B       | 1.9         | 1.9         | 0.119 | 55%          | Highest success rate |
| C       | 1.3         | 1.6         | 0.091 | 71%          | Lowest error at ≥70% success (local test) |
| C       | 1.6         | 1.6         | 0.095 | 76%          | Best balance                 |
| C       | 0.6         | 1.7         | 0.102 | 80%          | Highest success rate         |
| D       | 1.8         | 1.7         | 0.086 | 32%          | Lowest error at ≥30% success |
| D       | 1.5         | 1.3         | 0.101 | 69%          | Best balance (local test)    |
| D       | 1.0         | 0.7         | 0.102 | 82%          | Highest success rate         |


## Graph Architecture: SynthGraph

The codebase uses `SynthGraph` (defined in `graphs/synth_graph.py`) as its
primary graph representation. It wraps a **NetworKit** `nk.Graph` (C++ engine)
together with a NumPy positions array, replacing the previous `nx.Graph`.

```
SynthGraph
  ├── nk.Graph          # graph structure + edge weights (C++ backed)
  └── np.ndarray (N,2)  # node positions indexed by integer node ID
```

**Key methods**: `positions()`, `degree()`, `weighted_degree()`, `neighbors()`,
`edges()`, `edges_with_weights()`, `weight()`, `set_weight()`, `is_weighted()`,
`make_weighted()`, `make_unweighted()`, `largest_connected_component()`,
`subgraph()`, `copy()`, `from_networkx()`, `from_sparse_matrix()`,
`from_edge_list()`, `from_graph_nodes()`.

Whether a graph is weighted depends on where it came from, and it matters:
`from_sparse_matrix()` always produces a weighted graph, while
`from_edge_list()` and `from_graph_nodes()` produce unweighted ones.
`read_graph_csv()` follows the file — weighted only if the edge list has a
weight column. A generated synthetic network inherits its original's answer,
because the `Mapper` assigns widths only when the original had them.

### Where NetworkX is still used

NetworkX is **not** a dependency and is not in `requirements.txt`. It is
imported lazily in two places, and only to read pickles written before the
NetworKit migration:

| File | Purpose |
| ---- | ------- |
| `graphs/synth_graph.py` | `from_networkx()` — converts a legacy `nx.Graph` to `SynthGraph` |
| `graphs/graph_loader.py` | Detects a legacy `nx.Graph` payload in a `.pkl` and converts it |

Reading such a pickle requires `pip install networkx`; the loader says so
explicitly if it hits one. Nothing else needs it — all graph algorithms
(Dijkstra, betweenness, closeness, eigenvector centrality, diameter, connected
components, clustering) use **NetworKit** natively.

## Data Loader Requirements


| Function                            | Return Type            | Requirements                                              |
| ----------------------------------- | ---------------------- | --------------------------------------------------------- |
| `load_original_network(dataset_id)` | `SynthGraph`           | Built via `build_graph()` from positions + adjacency data |
| `load_original_image(dataset_id)`   | `np.ndarray` or `None` | CV2-compatible grayscale                                  |


## Log Analysis with `jq`

Each run writes its own structured **JSON lines** log to `run.jsonl` inside that run's output directory, so one run's log is never interleaved with another's. Each entry contains:

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
| `run_id` | 8-char hex ID unique to each run (safe for concurrent appends) |
| `level` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `logger` | Python module that emitted the record |
| `message` | Human-readable message |
| `tag` | *(optional)* Phase or category — `PHASE1`, `PHASE2`, `SNAPSHOT`, `IO`, `MEMORY`, `CONFIG`, `PIPELINE` |
| `dataset` | *(optional)* Dataset identifier, e.g. `sample_A` |

### Common queries

```bash
# Set the log file — latest_result points at the most recent run
LOG=data/output/latest_result/run.jsonl

# Show all entries from a specific run
jq 'select(.run_id == "304868d0")' "$LOG"

# Filter by tag (e.g. only Phase 1 events)
jq 'select(.tag == "PHASE1")' "$LOG"

# Show errors only
jq 'select(.level == "ERROR")' "$LOG"

# Errors with their tracebacks
jq 'select(.level == "ERROR") | {message, exception}' "$LOG"

# Messages for a specific dataset
jq 'select(.dataset == "sample_A") | .message' "$LOG"

# Combine filters: Phase 2 warnings for a dataset
jq 'select(.tag == "PHASE2" and .dataset == "sample_A" and .level == "WARNING")' "$LOG"

# Memory usage entries
jq 'select(.tag == "MEMORY") | "\(.ts) \(.message)"' "$LOG"

# List all run IDs in a file
jq -s '[.[].run_id] | unique' "$LOG"

# Count entries per tag
jq -s 'group_by(.tag) | map({tag: .[0].tag, count: length})' "$LOG"

# Follow a live run (stream mode)
tail -f "$LOG" | jq --unbuffered 'select(.tag == "PHASE1")'
```


## Troubleshooting

**Missing Input Files**

- Check file paths match your `BASE_INPUT_PATH` and dataset IDs

**Dimension Mismatch**

- Node positions shape `(N, 2)` must match adjacency matrix `(N, N)`

**Generation Failures**

```python
CLOSED_NODES_FACTOR *= 1.2  # Allow more node merging
ERROR_TOLERANCE *= 1.5      # Accept less similar networks
MAX_ATTEMPTS = 20           # More retry attempts
```

**Config Won't Load**

- Each config file must hold **exactly one** class defining `DATASETS`. The
  loader finds it by that attribute, not by name, and says so if a file has none
  or several.
- The class must set `MODE`, or `initialize()` raises: without it no pipeline
  claims the config.
- `SNAPSHOT_INTERVAL` needs saving enabled — snapshots bypass the `Saver`, so
  pairing it with `DISABLE_SAVING` raises rather than silently writing nothing.


## CI / Code Quality

Every push and pull request to `main` is checked by a **GitHub Actions** workflow (`.github/workflows/pre-commit.yml`) that runs the project's [pre-commit](https://pre-commit.com/) hooks:

| Hook | What it does |
| --- | --- |
| `trailing-whitespace` | Strips trailing whitespace |
| `end-of-file-fixer` | Ensures files end with a newline |
| `check-yaml` | Validates YAML syntax |
| `check-added-large-files` | Blocks files > 100 MB in `data/input/samples/`, default limit elsewhere |
| **Black** | Auto-formats Python code |
| **isort** | Sorts imports (Black-compatible profile) |
| **Flake8** | PEP 8 linting (`max-line-length=120`) |

To run the same checks locally before pushing:

```bash
pip install pre-commit
pre-commit install          # one-time setup – hooks run on every git commit
pre-commit run --all-files  # manual full check
```
