# Graph Network Analysis and Synthesis Toolkit

![Network Analysis](https://img.shields.io/badge/network-analysis-blue)
![Synthetic Generation](https://img.shields.io/badge/synthetic-generation-green)

A toolkit for analyzing real-world networks, extracting their structural properties, and generating synthetic networks with similar characteristics.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/WilliamLuminary/NetworkSynth.git
cd NetworkSynth
```

2. Install dependencies (Compatible with Python version `3.10.15`):

```bash
pip install -r requirements.txt
```

## Configuration

Edit `src/config/config_sample.py`:

```python
# Example configuration
class ConfigSample(Config):
    SETS = [SetName.Sample1, SetName.Sample2]
    RESOLUTIONS = [Resolution.HD]
  
    # Network generation parameters
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    SYNTHETIC_GRAPH_NUMBER = 500
  
    # Path configurations
    BASE_INPUT_PATH = './data/input'
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'positions')
```

## Project Structure

```
graph-network-toolkit/
├── src/
│   ├── config/             # Configuration management
... ...
│   └── main.py             # Entry point
├── data/
│   ├── input/              # Sample input data
│   └── output/             # Generated results
└── requirements.txt        # Dependencies
```

## Usage

All configurations should be done in the Config file used, simply run the `main.py` file.

```bash
python src/main.py
```
## Input Data Requirements

1. **Position Data**
   
   - Position of each node (x, y), should be a python list or numpy array
   - Shape: (N, 2) for N nodes
2. **Adjacency Matrix**
   
   - Sparse matrix should be a python list or numpy array
   - 
3. **Base Images**
   
   - As long as cv2.imread() is compatibl

Example input structure:

```
data/input/
├── positions/
│   └── Sample1_pos.npy
├── adjacency/
│   └── Sample1_mat.npy
└── images/
    └── Sample1_image.tif
```

## Outputs

Generated files include:

- Original network properties (PKL)
- Synthetic networks (PKL)
- Visualization images (PNG)
- Analysis reports (TXT)

```
results_[datetime]/    (Example)
├── [set_name]/
│   ├── [resolution]/
│   │   ├── 
│   │   ├── properties/
│   │   └── images/
...
└── latest_result -> results_[datetime]
```
