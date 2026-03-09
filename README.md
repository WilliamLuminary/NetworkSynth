# Graph Network Analysis and Synthesis Toolkit

Network Analysis
Synthetic Generation

## Quick Start

1. **Install and Run**
  ```bash
   git clone https://github.com/WilliamLuminary/NetworkSynth.git
   cd NetworkSynth
   pip install -r requirements.txt  # Python 3.10+

   python src/main.py
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
       │   ├── original_graph_*.png
       │   ├── original_image_*.png
       │   ├── original_network_*.pkl
       │   └── original_property_*.pkl
       └── synthetic/
           ├── len_10_err_0.123_synthetic_network_*.pkl
           └── synthetic_graph_*.png
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

1. Create `src/configs/generate_mode/config_mydata.py`:

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

1. Update `src/main.py`:

```python
from config.generate_mode import GenConfigMydata as GenConfig
GenConfig.initialize()
```

The config is automatically exported as `GenConfigMydata` based on the naming convention.

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
| C       | 1.3         | 1.6         | 0.091 | 71%          | Lowest error at ≥70% success |
| C       | 1.6         | 1.6         | 0.095 | 76%          | Best balance                 |
| C       | 0.6         | 1.7         | 0.102 | 80%          | Highest success rate         |
| D       | 1.8         | 1.7         | 0.086 | 32%          | Lowest error at ≥30% success |
| D       | 1.5         | 1.3         | 0.101 | 69%          | Best balance                 |
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

The codebase uses `SynthGraph` (defined in `src/graphs/synth_graph.py`) as its
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


| File                                        | Purpose                                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `src/graphs/synth_graph.py`                 | `from_networkx()` — converts legacy `nx.Graph` pickle files to `SynthGraph`                                    |
| `src/configs/analyze_mode/config_sample.py` | Detects old `.pkl` files containing `nx.Graph` and converts them via `SynthGraph.from_networkx()`              |
| `src/analysis/multifractal_analyzer.py`     | `to_networkx()` — converts back to `nx.Graph` only for `GraphRicciCurvature` (which requires `nx.Graph` input) |


All graph algorithms (Dijkstra, betweenness, closeness, eigenvector
centrality, diameter, connected components, clustering) now use **NetworKit**
natively.

## Data Loader Requirements


| Function                            | Return Type            | Requirements                                              |
| ----------------------------------- | ---------------------- | --------------------------------------------------------- |
| `load_original_network(dataset_id)` | `SynthGraph`           | Built via `build_graph()` from positions + adjacency data |
| `load_original_image(dataset_id)`   | `np.ndarray` or `None` | CV2-compatible grayscale                                  |


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
