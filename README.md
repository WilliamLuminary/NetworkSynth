# Graph Network Analysis and Synthesis Toolkit

Network Analysis
Synthetic Generation

## Quick Start

1. **Install and Run**
  ```bash
   git clone https://github.com/WilliamLuminary/NetworkSynth.git
   cd NetworkSynth
   pip install -r requirements.txt  # Python 3.14

   python run.py configs/generate_mode/config_sample.py
  ```

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
       │   ├── original_network_*.graphml.gz
       │   ├── original_property_*.json
       │   └── report.txt
       └── synthetic/
           ├── synthetic_graph_*.webp
           ├── synthetic_network_*.graphml.gz
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
| `save_json` | `.json` | any JSON-shaped object (numpy scalars included) |
| `save_pickle` | `.pkl` | any Python object — no longer used by any shipped spec |
| `save_csv` | `.csv` | rows (list of lists) |
| `save_text` | `.txt` | a string |
| `save_webp` | `.webp` | matplotlib Figure **or** ndarray **or** PIL image |
| `save_png` | `.png` | matplotlib Figure **or** ndarray **or** PIL image |
| `save_svg` | `.svg` (vector) | matplotlib Figure only |
| `save_network_csv` | `_edgelist.csv` + `_positions.csv` | a SynthGraph |
| `save_network_graphml` | `.graphml` or `.graphml.gz` | a SynthGraph — positions travel inside the file |

Rendered visual output comes in exactly two canonical kinds: a **matplotlib `Figure`** (the only kind that can also be saved as vector `.svg`) or a **BGR `ndarray`** (raster). `save_webp`/`save_png` accept either.

`save_network_csv` writes the `edge_weight` column **only for a weighted graph**.
An unweighted graph reports every weight as `1.0`, so writing the
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
| `original_network` | `SAVE_ORIGINAL_NETWORK` | `original/` | `.csv` **and** `.graphml.gz` |
| `original_property` | `SAVE_ORIGINAL_PROPERTY` | `original/` | `.json` |
| `original_report` | `SAVE_ORIGINAL_REPORT` | `original/` | `.txt`, no timestamp |
| `synthetic_graph` | `SAVE_SYNTHETIC_GRAPH` | `synthetic/` | `.webp` |
| `synthetic_network` | `SAVE_SYNTHETIC_NETWORK` | `synthetic/` | `.csv` **and** `.graphml.gz` |
| `synthetic_report` | `SAVE_SYNTHETIC_REPORT` | `synthetic/` | `.txt`, no timestamp |
| `analysis_data` | `SAVE_ANALYSIS_DATA` | root | `.json`, no timestamp |
| `analysis_figure` | `SAVE_ANALYSIS_FIGURE` | root | `.webp` |

`save_svg` (vector) and the uncompressed `.graphml` are **not** defaults but
remain available — name them in an override.

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
    save_network_graphml,
    save_png,
    save_svg,
    save_webp,
)


class ConfigDickson(BaseConfig):

    # Vector SVG instead of the default WebP:
    SAVE_ORIGINAL_GRAPH = (
        SaveSpec(ORIGINAL_DIR, "original_graph", "svg", save_svg),
    )

    # One save call, two files — the CSV pair and uncompressed GraphML:
    SAVE_SYNTHETIC_NETWORK = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "csv", save_network_csv),
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "graphml", save_network_graphml),
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
SAVE_MY_CUSTOM_DATA = (SaveSpec("custom_dir", "my_data", "json", save_json),)
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

An input is a directory of networks, a single `*_edgelist.csv`, or a single
`*.graphml` / `*.graphml.gz`. Each input is analysed as **its own labelled set**, so several inputs
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
analysis_data.json       # {results: {label: [...]}, measure_weighted, full_q_band}
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
frame side = max(d, MIN_CENTER_DISTANCE_FACTOR * max(FRAME_SIZE)) * TILE_FRAME_FACTOR
```

The floor is the same rule at the closest spacing the sampler will allow, so it is measured rather than configured — there is no pixel count to keep in step with the frame. Centers are never placed closer than `MIN_CENTER_DISTANCE_FACTOR * max(FRAME_SIZE)` apart, which makes the floor a guard rather than something a run normally meets.

Tiles in crowded regions get smaller frames and tiles in open space get larger ones — each scaled to the room it actually has.

| Parameter | Default | Description |
| --- | --- | --- |
| `TILE_FRAME_SIZE` | `None` | Fixed frame `(w, h)` for every tile. `None` enables auto mode. |
| `TILE_FRAME_FACTOR` | `0.5` | Auto mode: frame side as a fraction of the nearest-neighbor distance. Lower = wider seam for Phase 2 to stitch; higher = tiles nearly touch. |

The Phase 1 log (tag `PHASE1`) prints the resulting frame-side min/max for each run, so you can confirm tiles aren't being shrunk below a usable size.

Where the centers land follows the run's `SEED`, like everything else a run draws: a seed replays the same scatter, `None` scatters differently every time.


## GUI vs CLI

`gui_run.py` and `run.py` end in the same two lines — `load_pipeline(config.MODE)` then `pipeline.main(config_cls=config)` — so a GUI run and a CLI run execute identical pipeline, generator, quality-gate and save code. They differ only in where the config comes from: a `configs/*_mode/config_*.py` module, or a `GuiConfig` subclass built from the run-spec the form writes.

### A directory of networks

The GUI can be pointed at a folder instead of a single network, and reads one dataset per **prefix**. Which files carry a prefix decides what format that dataset is in, so one folder may hold a mixture:

| Files | The dataset is |
| --- | --- |
| `<prefix>_edgelist.csv` + `<prefix>_positions.csv` | a CSV pair |
| `<prefix>_adjacency.npy` + `<prefix>_positions.npy` | a NumPy pair, positions transposed on load |
| `<prefix>_network.graphml` or `.graphml.gz` | one network, positions carried inside the file |
| `<prefix>_image.tif` | the background for that prefix — optional, and independent of the three above |

A half-named dataset stops the run rather than being passed over: an edge list with no positions beside it, an adjacency with no coordinates, or one prefix named as two datasets at once. Datasets are read in name order, and one run writes a subdirectory per dataset under a single run root with one manifest covering them all.

Two things this does *not* cover. The npy files shipped in `data/input/samples/` keep the older `_mat.npy` / `_pos.npy` names that the CLI configs load directly, so those folders are not directory-discoverable. And `analyse.py` reads a directory by its own rule — every GraphML file in it, or else every CSV pair — not by this contract.

The form is a deliberate subset. What only a CLI config can do:

| Only on the CLI | Why it matters |
| --- | --- |
| `mosaic`, `scaling`, `compare` | The GUI offers `generate`, `hybrid` and `sweep`; the other three pipelines are CLI-only. |
| `DATASET_FACTORS` | Per-dataset `(CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR)` overrides. A GUI directory run applies one pair to every dataset in the folder. |
| `TILE_FRAME_SIZE` | Fixed tile frames. Every shipped hybrid config uses auto sizing, so the form offers auto only. |
| `SNAPSHOT_INTERVAL < 0` | Log-spaced snapshots (~\|N\| in total, hybrid only). The form takes a positive interval and a switch. |
| `.svg` for a hybrid run | The GUI omits that switch in Hybrid: the assembled image and its snapshots are drawn as pixels by OpenCV, so there is no vector to write. A CLI config that puts `save_svg` in `SAVE_SYNTHETIC_GRAPH` for hybrid fails when it saves. |
| `PHASE2_MAX_ROUNDS`, `SELECT_BEST`, `MIN_TILE_NODES`, `LOG_MEMORY`, `DISABLE_SAVING` | Left at their `BaseConfig` defaults for a GUI run. |

The other way round, `INPUT_ORIENTATION` — a quarter turn applied to the input network as it is read — belongs to the form's Align card and has no CLI equivalent; CLI configs transpose inside their own loaders instead.

Drawing is shared but not single-source: matplotlib output goes through `plot_network` (runs and previews alike) and `save_bfs_snapshot`; OpenCV output through `render_network` and `save_hybrid_snapshot`. The GUI adds no network renderer of its own — the preview builds only its background figure.


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
primary graph representation. It wraps an **igraph** `Graph` (C engine) together
with a NumPy positions array.

```
SynthGraph
  ├── igraph.Graph      # graph structure, edge weights in es["weight"]
  └── np.ndarray (N,2)  # node positions indexed by integer node ID
```

**Key methods**: `positions()`, `degree()`, `weighted_degree()`, `neighbors()`,
`edges()`, `edges_with_weights()`, `weight()`, `set_weight()`, `is_weighted()`,
`make_weighted()`, `make_unweighted()`, `largest_connected_component()`,
`subgraph()`, `copy()`, `component_sizes()`, `local_clustering()`,
`from_sparse_matrix()`, `from_edge_list()`, `from_graph_nodes()`.

While the migration off NetworKit finishes, `SynthGraph.nk` converts to a
NetworKit graph on demand for `analysis/multifractal_analyzer.py`, the last
consumer that has not moved. It is temporary and goes away with that file.

Whether a graph is weighted depends on where it came from, and it matters:
`from_sparse_matrix()` always produces a weighted graph, while
`from_edge_list()` and `from_graph_nodes()` produce unweighted ones.
`read_graph_csv()` follows the file — weighted only if the edge list has a
weight column. A generated synthetic network inherits its original's answer,
because the `Mapper` assigns widths only when the original had them.

### NetworkX is not used

NetworkX is **not** a dependency, is not in `requirements.txt`, and is no longer
imported anywhere. It used to be reached for lazily to rebuild a legacy
`nx.Graph` out of a pickle; both that loader and the `.pkl` input form are gone.

All graph algorithms — shortest paths, betweenness, closeness, eigenvector
centrality, diameter, connected components, clustering — run on **igraph**.

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


## The `dist` Branch

`dist` is a code-only copy of this repository, published so another project can carry NetworkSynth as a git submodule without dragging in 100 MB of tracked data. It is an **orphan branch** — its history has no ancestor in `main` — and its trees hold only the seven packages, the four entry points, `requirements.txt` and a README of its own. `.gitignore` ships too, so that once a consumer makes a `.venv` inside the checkout and runs something, their `git status` does not report the submodule as dirty. A shallow clone of it comes to about 780 KB — 544 KB of files and a 232 KB `.git` — against 162 MB for a full clone of this repository.

Nothing is edited on `dist` directly. Every commit there is generated:

```bash
scripts/publish_dist.sh [source-ref] [tag]     # defaults to HEAD, no tag
```

The script reads the allowlisted paths out of the *source commit* through a temporary index and writes the result with `commit-tree`. The working tree is never read, so nothing untracked or ignored — `data/output`, `wandb/`, `.venv` — can reach the branch, and a publish works fine from a dirty branch or from an old tag. It stops with the name of anything on the allowlist that is missing from the source ref, says so and exits `0` when the tree is unchanged since the last publish, and never pushes.

Only the first publish is an orphan commit; later ones are its children, so `dist` has an ordinary linear history. That is what lets a shallow submodule fetch and a tag between releases work without force-pushing.

### The two READMEs

`README.md` is this file, for people working *on* NetworkSynth. `README_dist.md` is for people who found NetworkSynth inside someone else's project: setup, the three entry points, the input contract, and where the full documentation lives. The publish script maps it to `README.md` on the branch — the `README_dist.md:README.md` entry in its `PATHS` list, which is the general form for shipping any path under another name.

Keeping both here rather than committing a README onto `dist` is deliberate: a file edited on the branch would be overwritten by the next publish.

### Releasing

Tag `main` and push the tag. That is the whole release:

```bash
git tag -a v1.0.1 -m "NetworkSynth v1.0.1"
git push origin v1.0.1
```

`.github/workflows/publish-dist.yml` picks that up, runs the publish script against the tagged commit, and pushes `dist` along with a matching `dist-v1.0.1`. Three tag shapes, each meaning one thing:

| Tag | On | Made by |
| --- | --- | --- |
| `v1.0.1` | `main` | you, and it is what triggers everything |
| `dist-v1.0.1` | `dist` | the workflow, one per `v` tag |
| `submission-v1.0.0` | a code-deposit branch | you, for a journal submission — unrelated to either of the above |

`dist-v*` deliberately does not match the workflow's `v*` trigger, so publishing cannot set itself off again; pushes made with `GITHUB_TOKEN` do not start workflows either, so that holds twice over.

#### What the workflow does

Six steps, in `.github/workflows/publish-dist.yml`:

1. **Check out the tagged commit**, with `fetch-depth: 0` — the job needs `origin/main` to vet the tag and the `dist` branch to build on, neither of which a shallow checkout brings.
2. **Refuse a tag that is not on main**, with `git merge-base --is-ancestor "$GITHUB_SHA" origin/main`. A `v` tag pushed from a feature branch fails the job instead of quietly publishing from it.
3. **Set the committer** to `github-actions[bot]`, so the generated commit is attributable to the automation rather than to whoever tagged.
4. **Bring `dist` into the checkout.** `actions/checkout` leaves only remote-tracking refs, and the publish script looks for the local branch `refs/heads/dist`, so the job creates it from `origin/dist` — and skips that on the very first release, when the branch does not exist anywhere yet.
5. **Publish**, with `scripts/publish_dist.sh "$GITHUB_SHA" "dist-$GITHUB_REF_NAME"`. The dist tag's version is derived from the release tag rather than chosen, so `v1.0.1` yields `dist-v1.0.1` and the two can never drift apart.
6. **Push** `dist` and the new tag in one command.

The job holds a `publish-dist` concurrency group, so two releases pushed close together queue rather than race to write the same branch, and it asks for `contents: write` — supplied by the default `GITHUB_TOKEN`, with no secret to configure.

Two behaviours worth expecting. A release whose code did not change still gets its `dist-` tag, on the `dist` commit already there: the script reports `already carries this tree` and tags that commit anyway, because a tag that silently failed to appear would strand anyone who went looking for it. And re-running the workflow for a tag that was already published fails at the tagging step, since `git tag` will not move an existing tag — which is the right outcome, as a published `dist-` tag is something a consumer may already have pinned.

To publish by hand — a dry run, or a `dist` commit with no release behind it:

```bash
scripts/publish_dist.sh HEAD             # no tag; prints the push command
git push origin dist
```

#### Taking a release, on the consumer side

A submodule records a commit, not a branch, so a project carrying NetworkSynth keeps building against exactly what it was pinned to until someone moves the pin. Publishing a release changes nothing for them until they do:

```bash
git -C third_party/NetworkSynth fetch --tags
git -C third_party/NetworkSynth checkout dist-v1.0.1
git add third_party/NetworkSynth && git commit -m "bump NetworkSynth to dist-v1.0.1"
```

The version they name is the one you tagged on `main` — that is the point of deriving `dist-v1.0.1` from `v1.0.1` rather than numbering the two branches separately. Committing that gitlink is what pins it: any later checkout of their repository brings back the same NetworkSynth it was built against, and a release of ours can never silently change what their application runs.

If a consumer would rather follow the tip of `dist` than name versions, `git submodule update --remote` does that — `.gitmodules` records `branch = dist`, so the pin moves to whatever was published last. That trades the guarantee above for one less step, which is usually the wrong trade.

`INTEGRATION_PLAN.md` covers the other side of this — how StructuralGT finds and launches us.


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
