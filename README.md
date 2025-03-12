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

### 1. Modify Key Parameters

Edit `src/config/config_sample.py`:

```python
class ConfigSample(Config):
    # Generation parameters
    CLOSED_NODES_FACTOR = 1.5    # Original: 1.2
    CLOSED_EDGES_FACTOR = 1.0    # Original: 0.8
    SYNTHETIC_NETWORK_NUMBER = 50# Original: 10
    ...
```

### 2. Common Parameters
| Parameter              | Typical Values               | Effect on Generation                   |
|------------------------|------------------------------|----------------------------------------|
| CLOSED_NODES_FACTOR    | 0.5-2.0                      | Controls node merging likelihood       |
| CLOSED_EDGES_FACTOR    | 0.5-2.0                      | Affects edge proximity tolerance       |
| ERROR_TOLERANCE        | 0.1-0.5                      | Multifractal similarity threshold      |
| SYNTHETIC_GRAPH_NUMBER | $\leq$ SYNTHETIC_NETWORK_NUM | Visualized network generated per batch |
| ...                    |                              |                                        |

## Advanced Configuration

### 1. Add New Dataset Type
```python
# src/config/enums.py
class SetName(Enum):
    MY_SET = "my_set"  # Add new dataset identifier
    
class Resolution(Enum):
    MY_RES = "my_res"  # Add new dataset identifier
    
# src/config/config_custom.py
class ConfigCustom(Config):
    SETS = [SetName.MY_SET]
    RESOLUTIONS = [Resolution.NA]
    # If no need for two levels of identifiers, use NA.
    
    @staticmethod
    def _load_positions(set_name, resolution):
        return np.load(f'data/{set_name}/positions.npy')
```

### 2. Implement Data Loaders
Required function signatures:
   ```python
   def _load_positions(set_name, resolution) -> np.ndarray:  # Shape: [N,2]
   def _load_sparse_matrix(set_name, resolution) -> dict:     # {'rows':[], 'cols':[]}
   def _load_image(set_name, resolution) -> np.ndarray:       # CV2-compatible format
   ```

#### Input Data Requirements

| Functions             | Return Type         | Return Requirements                              |
|-----------------------|---------------------|--------------------------------------------------|
| _load_positions()     | list or numpy array | Shape of (N, 2)                                  |
| _load_sparse_matrix() | (the same as above) | Recognized by networkx.from_scipy_sparse_array() |
| _load_image()         | (the same as above) | Recognized by cv2.imread()                       |


## Configuration Workflow

### 1. Choose Base Template

```python
# In src/main.py
ConfigSample.initialize()  # Simple template
# ConfigCustom.initialize()  # Custom configuration
```

### 2. Implement Data Loaders

Copy from any `config_[NAME].py` file, then only modify the **name of the class** and **values of variables**
(`_load_[DATA]()` functions are also required).

```python
class ConfigCustom(Config):
    @staticmethod
    def _load_positions(set_name, resolution):
        """Custom position loader for your data"""
        file_path = f"data/positions/{set_name}_nodes.npy"
        return np.load(file_path)
    ... # implement other 
```
## Output Interpretation

### File Type Mapping
| Pattern                   | Data Type            | Visualization Example    |
|---------------------------|----------------------|--------------------------|
| `original_graph_*.png`    | Network topology     | [Sample Graph]           |
| `original_property_*.pkl` | Degree distributions | {2: 0.6, 3: 0.3, 4: 0.1} |
| `synthetic_network_*.pkl` | Generated network    | NetworkX Graph object    |


## Troubleshooting

### Common Issues
1. **Missing Input Files**  
   Ensure files follow naming convention:  
   `{SetName}_pos.npy`, `{SetName}_mat.npy`, `{SetName}_image.tif`

2. **Dimension Mismatch**  
   Verify node positions `(N,2)` match adjacency matrix `(N,N)`

3. **Generation Failures**  
   Adjust thresholds:
   ```python
   CLOSED_NODES_FACTOR *= 1.2  # Allow more node merging
   ERROR_TOLERANCE *= 1.5       # Accept less similar networks
   ```