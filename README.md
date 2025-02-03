# Graph Network Analysis and Synthesis Toolkit

![Network Analysis](https://img.shields.io/badge/network-analysis-blue)
![Synthetic Generation](https://img.shields.io/badge/synthetic-generation-green)

## Quick start

1. Run with Default Configuration

```bash
# Clone and install
git clone https://github.com/WilliamLuminary/NetworkSynth.git
cd NetworkSynth
# Compatible with Python version 3.10.15
pip install -r requirements.txt

# Run with sample configuration
python src/main.py
```

2. Prepare Sample Data

Create this structure in data/sample_input:

```bash
sample_input/
├── sample_1_pos.npy    # Node positions
├── sample_1_mat.npy    # Adjacency matrix
└── sample_1_image.tif  # Base image
```

3. Expected Output

```bash
results_YYYYMMDD_HHMM/
├── sample_1/
│   (no resolution dir since Resolution.NA)
│   ├── origin/
│   │   ├── original_graph_YYYYMMDD_HHMM.png
│   │   ├── original_image_YYYYMMDD_HHMM.png
│   │   ├── original_network_YYYYMMDD_HHMM.pkl
│   │   └── original_property_YYYYMMDD_HHMM.pkl
│   ├── synthetic/
│   │   ├── nw_no_10_err_0.123_synthetic_network_YYYYMMDD_HHMM.pkl
│   │   ├── synthetic_graph_YYYYMMDD_HHMM.png
│   │   ├── synthetic_graph_YYYYMMDD_HHMM.png
│   │   └── synthetic_graph_YYYYMMDD_HHMM.png
```

## Basic Customization <small>(No Code Changes)</small>

### Core Enums Configuration

```python
# Defined in src/config/enums.py
class SetName(Enum):
    # Legacy microscopy sets
    A = "A"        # 10Kx sample group A
    B = "B"        # 10Kx sample group B
    # New microscopy sets (W- series)
    S4  = "004"    # W-2-89-1_004 sample
    S8  = "008"    # W-2-89-1_008 sample
    # Sample datasets
    Sample1 = "sample_1"  # Demonstration set 1
    NA = ""                # Not applicable

class Resolution(Enum):
    X10K = "10kX"   # 10,000x magnification
    X20K = "20kX"   # 20,000x magnification
    NA = ""          # Not applicable
```

### Extending Enums for Custom Data

To add new datasets/resolutions:

1. **Modify SetName Enum**
   Add new entries for your dataset identifiers:

   ```python
   class SetName(Enum):
       EXPERIMENT1 = "exp1"  # New experimental set
       EXPERIMENT2 = "exp2"
       # ... existing entries
   ```
2. **Modify Resolution Enum** (if needed)
   Add new microscopy resolutions:

   ```python
   class Resolution(Enum):
       X25K = "25kX"  # New resolution level
       # ... existing entries
   ```
3. **Update Configuration**
   Use your new enums in custom configurations:

   ```python
   class ConfigExperiment(Config):
       SETS = [SetName.EXPERIMENT1, SetName.EXPERIMENT2]
       RESOLUTIONS = [Resolution.X25K]
   ```

## Input Data Structure

### File Naming Convention


| Component        | Pattern               | Example          |
| ---------------- | --------------------- | ---------------- |
| Position Data    | `{SetName}_pos.npy`   | `exp1_pos.npy`   |
| Adjacency Matrix | `{SetName}_mat.npy`   | `exp1_mat.npy`   |
| Base Image       | `{SetName}_image.tif` | `exp1_image.tif` |

### Directory Organization

```bash
data/custom_input/
├── exp1_pos.npy          # Node coordinates
├── exp1_mat.npy          # Sparse adjacency matrix
├── exp1_image.tif        # Source microscopy image
├── exp2_pos.npy
├── exp2_mat.npy
└── exp2_image.tif
```

## Configuration Workflow

### 1. Choose Base Template

```python
# In src/main.py
ConfigSample.initialize()  # Simple template
# ConfigCustom.initialize()  # Custom configuration
```

### 2. Implement Data Loaders

```python
class ConfigCustom(Config):
    @staticmethod
    def _load_positions(set_name, resolution):
        """Custom position loader for experimental data"""
        file_path = f"data/positions/{set_name}_nodes.npy"
        return np.load(file_path)
```

### 3. Directory Handling

NA values in SetName/Resolution flatten the output structure:

```
results_20231125_1430/
├── exp1/                # SetName.EXPERIMENT1
│   └── graphs/          # No resolution subdirectory (Resolution.NA)
└── exp2/
    └── properties/
```

## Key Configuration Parameters


| Parameter              | Typical Values               | Effect on Generation                   |
| ---------------------- | ---------------------------- | -------------------------------------- |
| CLOSED_NODES_FACTOR    | 0.5-2.0                      | Controls node merging likelihood       |
| CLOSED_EDGES_FACTOR    | 0.5-2.0                      | Affects edge proximity tolerance       |
| ERROR_TOLERANCE        | 0.1-0.5                      | Multifractal similarity threshold      |
| SYNTHETIC_GRAPH_NUMBER | $\leq$ SYNTHETIC_NETWORK_NUM | Visualized network generated per batch |

| SYNTHETIC_NETWORK_NUM | $\geq$ SYNTHETIC_GRAPH_NUMBER | Networks generated per batch        |

## Output Structure

### Standard Output Hierarchy

```
results_20231125_1430/               # Timestamped results
├── Sample1/                         # SetName.Sample1
│   └── NA/                          # Resolution.NA
│       ├── original/
│       │   ├── original_graph_20231125_1430.png     # Network visualization
│       │   ├── original_image_20231125_1430.png     # Processed base image
│       │   ├── original_network_20231125_1430.pkl   # NetworkX graph
│       │   └── original_property_20231125_1430.pkl  # Graph attributes
│       └── synthetic/
│           ├── synthetic_graph_20231125_1430.png    # Generated network visual
│           └── synthetic_network_20231125_1430.pkl  # Synthetic graph data
└── latest_result -> results_20231125_1430           # Symlink to latest
```

### NA Resolution Handling Example

When using `Resolution.NA`:

```
results_20231125_1430/
└── Sample1/              # Skips resolution directory
    ├── original/
    └── synthetic/
```

### Key File Types


| File Pattern              | Content Type               | Format |
| ------------------------- | -------------------------- | ------ |
| `original_graph_*.png`    | Network visualization      | PNG    |
| `original_image_*.png`    | Processed microscopy image | PNG    |
| `original_network_*.pkl`  | NetworkX graph object      | Pickle |
| `original_property_*.pkl` | GraphAttrAgent analysis    | Pickle |
| `synthetic_graph_*.png`   | Generated network visual   | PNG    |
| `synthetic_network_*.pkl` | Synthetic graph data       | Pickle |

## Configuration Parameter Clarification

### Key Generation Parameters


| Parameter                  | Relationship                | Typical Value | Description                        |
| -------------------------- | --------------------------- | ------------- | ---------------------------------- |
| `SYNTHETIC_GRAPH_NUMBER`   | ≤ SYNTHETIC_NETWORK_NUMBER | 50            | Number of visualized networks      |
| `SYNTHETIC_NETWORK_NUMBER` | ≥ SYNTHETIC_GRAPH_NUMBER   | 300           | Total networks generated per batch |
| `ERROR_TOLERANCE`          | Independent                 | 0.15          | Multifractal similarity threshold  |
