# Graph Network Analysis and Synthesis Toolkit

Network Analysis
Synthetic Generation

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

## Quick Start

1. **Install and Run**
  ```bash
   git clone https://github.com/WilliamLuminary/NetworkSynth.git
   cd NetworkSynth
   pip install -r requirements.txt  # Python 3.10+

   python -m run generate
  ```
2. **Prepare Sample Data**
  ```
   data/input/samples/generate_mode/
   ├── sample_1_pos.npy    # Node positions (N, 2)
   ├── sample_1_mat.npy    # Adjacency matrix
   └── sample_1_image.tif  # Background image (optional)
  ```
3. **Expected Output**
  ```
   data/output/ConfigName_results_YYYYMMDD_HHMMSS/
   └── sample_1/
       ├── original/
       │   ├── original_graph_*.svg
       │   ├── original_image_*.png
       │   ├── original_network_*_edgelist.csv
       │   ├── original_network_*_positions.csv
       │   ├── original_network_*.nkbin
       │   ├── original_network_*_positions.npy
       │   └── original_property_*.pkl
       └── synthetic/
           ├── synthetic_graph_*.webp
           ├── synthetic_network_*.pkl
           ├── synthetic_network_*_edgelist.csv
           ├── synthetic_network_*_positions.csv
           ├── synthetic_network_*.nkbin
           └── synthetic_network_*_positions.npy
  ```

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
from ..enums import DatasetId

def _generate_datasets() -> List[DatasetId]:
    return [DatasetId("set_1"), DatasetId("set_2")]

class ConfigMydata(BaseConfig):
    DATASETS = _generate_datasets()

    IMAGE_SIZE = (1024, 1024)
    FRAME_SIZE = (512, 512)
    SYNTHETIC_FRAME_SIZE = (1536, 1536)

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    SYNTHETIC_NETWORK_NUMBER = 10
    SYNTHETIC_GRAPH_NUMBER = 3
    ERROR_TOLERANCE = 0.15

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

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

2. Run with your config:

```bash
python -m run generate --config Mydata
```

The config is automatically exported as `GenConfigMydata` based on the naming convention.

### Save System

Saving is config-driven: each config declares **what** formats to save, and the `Saver` handles **how** (path construction, timestamps, directory creation).

#### Three layers

```
Pipeline  →  Saver.save(content, identifier, prefix)
                 ↓
              Config.save(identifier)  →  returns list of specs
                 ↓
              Serializer(content, filepath)  →  writes to disk
```

**Layer 1 — Serializers** (`configs/file_definitions.py`): Pure `(content, filepath)` functions.

| Serializer | Output |
| --- | --- |
| `save_pickle` | `.pkl` |
| `save_csv` | `.csv` |
| `save_png` | `.png` (matplotlib Figure or numpy array) |
| `save_svg` | `.svg` (matplotlib Figure) |
| `save_webp` | `.webp` (matplotlib Figure) |
| `save_networkit` | `.nkbin` + companion `_positions.npy` |
| `save_network_csv` | `_edgelist.csv` + `_positions.csv` (from a SynthGraph) |
| `save_network_nkbin` | `.nkbin` + `_positions.npy` (from a SynthGraph) |

**Layer 2 — Config save methods** (`BaseConfig` + mode overrides): Each `save_*` classmethod returns a list of spec tuples — no dependency on the Saver.

**Layer 3 — Saver** (`handlers/saver.py`): Single public method `save()`. Gets specs from the config, builds file paths, creates directories, and calls the serializer with `(content, filepath)`.

#### How specs work

Each `save_*` method returns a list of tuples:

```python
(relative_dir, detail, extension, save_fn)
# Optional 5th element: False to disable timestamp
```

The Saver constructs the path as: `{output_dir}/{relative_dir}/{prefix}{detail}_{timestamp}.{extension}`

#### Default save methods

`BaseConfig` defines defaults for all standard identifiers. These are inherited by every mode config automatically:

| Identifier | Directory | Formats | Serializer |
| --- | --- | --- | --- |
| `original_image` | `original/` | `.png` | `save_png` |
| `original_network` | `original/` | `.csv`, `.nkbin` | `save_network_csv`, `save_network_nkbin` |
| `original_property` | `original/` | `.pkl` | `save_pickle` |
| `original_graph` | `original/` | `.svg` | `save_svg` |
| `synthetic_graph` | `synthetic/` | `.webp` | `save_webp` |
| `synthetic_network` | `synthetic/` | `.pkl` | `save_pickle` |
| `synthetic_export` | `synthetic/` | `.csv`, `.nkbin` | `save_network_csv`, `save_network_nkbin` |
| `analysis_data` | root | `.pkl` (no timestamp) | `save_pickle` |
| `analysis_figure` | root | `.svg` | `save_svg` |

#### Overriding save behavior

A mode config can override any `save_*` method to change output formats. For example, mosaic mode saves synthetic graphs in both webp and png:

```python
class SampleConfig(BaseConfig):
    @classmethod
    def save_synthetic_graph(cls):
        from ..file_definitions import save_png, save_webp

        return [
            ("synthetic", "synthetic_graph", "webp", save_webp),
            ("synthetic", "synthetic_graph", "png", save_png),
        ]
```

Only override what differs — everything else is inherited from `BaseConfig`.

#### Adding a new save format

To save a new type of data:

1. Add a constant to `DataType` in `configs/enums.py`:

```python
class DataType:
    MY_CUSTOM_DATA = "my_custom_data"
```

2. Add a `save_my_custom_data` method to `BaseConfig` (or your mode config):

```python
@classmethod
def save_my_custom_data(cls):
    from .file_definitions import save_pickle
    return [("custom_dir", "my_data", "pkl", save_pickle)]
```

3. Call it from the pipeline:

```python
data_agent.save(DataType.MY_CUSTOM_DATA, content=my_data)
```

### Config Naming Convention


| File Name             | Class Name        | Exported As          |
| --------------------- | ----------------- | -------------------- |
| `config_sample.py`    | `SampleConfig`    | `GenConfig`          |
| `config_nanowires.py` | `ConfigNanowires` | `GenConfigNanowires` |
| `config_mydata.py`    | `ConfigMydata`    | `GenConfigMydata`    |


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


## Key Parameters


| Parameter                  | Typical Values | Description                           |
| -------------------------- | -------------- | ------------------------------------- |
| `CLOSED_NODES_FACTOR`      | 0.5-2.0        | Node merging likelihood               |
| `CLOSED_EDGES_FACTOR`      | 0.5-2.0        | Edge proximity tolerance              |
| `ERROR_TOLERANCE`          | 0.1-0.5        | Multifractal similarity threshold     |
| `SYNTHETIC_NETWORK_NUMBER` | 1-100          | Networks to generate per dataset      |
| `SYNTHETIC_GRAPH_NUMBER`   | 0-10           | Graphs to visualize (≤ network count) |
| `MAX_ATTEMPTS`             | 5-20           | Retry attempts per network            |


## Graph Architecture: SynthGraph

The codebase uses `SynthGraph` (defined in `graphs/synth_graph.py`) as its
primary graph representation. It wraps a **NetworKit** `nk.Graph` (C++ engine)
together with a NumPy positions array, replacing the previous `nx.Graph`.

```
SynthGraph
  ├── nk.Graph          # graph structure + edge weights (C++ backed)
  └── np.ndarray (N,2)  # node positions indexed by integer node ID
```

**Key methods**: `positions()`, `degree()`, `neighbors()`, `edges()`,
`weight()`, `set_weight()`, `largest_connected_component()`, `subgraph()`,
`copy()`, `to_networkx()`, `from_networkx()`, `from_sparse_matrix()`,
`from_graph_nodes()`.

### Where NetworkX is still used

NetworkX (`networkx`) remains installed as a dependency but is only imported
in three specific places:


| File                                     | Purpose                                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------- |
| `graphs/synth_graph.py`                  | `from_networkx()` — converts legacy `nx.Graph` pickle files to `SynthGraph`      |
| `configs/analyze_mode/config_sample.py`  | Detects old `.pkl` files containing `nx.Graph` and converts via `from_networkx()` |
| `scripts/helpers/convert_pkl_network.py` | Standalone converter — reads `nx.Graph` pickles, exports CSV + NetworKit binary   |


All graph algorithms (Dijkstra, betweenness, closeness, eigenvector
centrality, diameter, connected components, clustering) now use **NetworKit**
natively.

## Data Loader Requirements


| Function                            | Return Type            | Requirements                                              |
| ----------------------------------- | ---------------------- | --------------------------------------------------------- |
| `load_original_network(dataset_id)` | `SynthGraph`           | Built via `build_graph()` from positions + adjacency data |
| `load_original_image(dataset_id)`   | `np.ndarray` or `None` | CV2-compatible grayscale                                  |


## Log Analysis with `jq`

All pipeline runs emit structured **JSON lines** (`.jsonl`) logs to `data/output/logs/`. Each log entry contains:

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
# Set the log file (tab-complete friendly)
LOG=data/output/logs/project_SampleConfig.jsonl

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

**Import Errors**

- Ensure config class follows naming convention: `ConfigXxx` in `config_xxx.py`
