# NetworkSynth

Generates synthetic networks modelled on a real one. It measures an input network's structure, grows candidates that match it, and keeps the ones that pass a quality gate.

## Quick start

```bash
git clone https://github.com/WilliamLuminary/NetworkSynth.git
cd NetworkSynth
pip install -r requirements.txt   # Python 3.14

python run.py configs/generate_mode/config_sample.py
```

That config reads from `data/input/samples/generate_mode/`:

```
sample_1_pos.npy      # node positions, shape (N, 2)
sample_1_mat.npy      # adjacency matrix
sample_1_image.tif    # background image, optional
```

Every run writes one directory under `data/output/`, and points `data/output/latest_result` at it:

```
generate_mode_SampleConfig_results_20260824_130049_c6a6a6ec/
├── manifest.json      # what the run produced, and how it ended
├── run.jsonl          # this run's log, one JSON object per line
└── sample_1/
    ├── original/      # the input network, its image, a report
    └── synthetic/     # the generated networks and their plots
```

Networks are saved as a CSV pair, `*_edgelist.csv` plus `*_positions.csv`. GraphML is supported but off by default.

`manifest.json` is written even when a run fails or is cancelled, so its absence means the process died hard. Setting `DISABLE_SAVING` writes nothing at all, not even the directory.

## Configs

A run takes one config file and nothing else. `run.py` with no argument lists them all.

Each file holds exactly one class with a `DATASETS` attribute. Its `MODE` picks the pipeline: `generate`, `hybrid`, `mosaic`, `scaling`, `sweep` or `compare`. Nothing needs registering, the file is the name.

### Writing one

```python
from ..base_config import BaseConfig
from ..dataset_id import DatasetId


class ConfigMydata(BaseConfig):
    MODE = "generate"
    DATASETS = [DatasetId("set_1"), DatasetId("set_2")]

    IMAGE_SIZE = (1024, 1024)
    FRAME_SIZE = (512, 512)
    SYNTHETIC_FRAME_SIZE = (1536, 1536)

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    SYNTHETIC_NETWORK_NUMBER = 10
    SYNTHETIC_GRAPH_NUMBER = 3
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        ...   # return a SynthGraph, usually via build_graph(positions, matrix)

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        ...   # return a grayscale numpy array, or None
```

`MEASURE_WEIGHTED` has no default on purpose. Measuring a weighted network as plain topology is a real choice, so it cannot be guessed from the data, and asking for widths a network does not carry raises immediately.

A `DatasetId` names one dataset and can have levels:

```python
DatasetId("1-0").path        # "1-0"
DatasetId("A", "10kX").path  # "A/10kX"
```

### Key parameters

| Parameter | Typical values | What it does |
| --- | --- | --- |
| `CLOSED_NODES_FACTOR` | 0.5 to 2.0 | How readily nodes merge |
| `CLOSED_EDGES_FACTOR` | 0.5 to 2.0 | Edge proximity tolerance |
| `ERROR_CHECKER` | `multifractal` or `none` | Quality gate. `none` skips checking |
| `ERROR_TOLERANCE` | 0.1 to 0.5 | How close a candidate must be to pass |
| `SYNTHETIC_NETWORK_NUMBER` | 1 to 100 | Networks generated per dataset |
| `SYNTHETIC_GRAPH_NUMBER` | 0 to 10 | How many of those get plotted |
| `MAX_ATTEMPTS` | 5 to 20 | Retries per network |

## Measuring networks you already have

Generating a network and measuring one are different jobs, so measuring has its own entry point and no config module:

```bash
python analyse.py                                  # uses the constants in the script
python analyse.py <input> [<input> ...] <out_dir>   # last argument is the output directory
```

An input is a directory of networks, one `*_edgelist.csv`, or one `*.graphml` or `*.graphml.gz`. Each input is measured as its own labelled set, so several inputs give one figure per measure with every set drawn on it. Labels come from the paths, taking in parent directories as needed to stay distinct.

Four constants at the top of the script hold the defaults, and the two paths can be overridden on the command line:

| Constant | Meaning |
| --- | --- |
| `INPUTS` | Default input paths |
| `OUTPUT_DIR` | Default output directory |
| `MEASURE_WEIGHTED` | Measure edge widths, or topology only |
| `FULL_Q_BAND` | Wide q band (401 points, `-20..20`) instead of the narrow one (61 points, `-3..3`) |

It writes a plain directory, with no run root and no manifest:

```
analysis_data.json        # {results: {label: [...]}, measure_weighted, full_q_band}
analysis_spectra.webp     # f(alpha) against alpha, one curve per network
analysis_dimensions.webp  # D(q) against q
```

The measurement settings live inside the data file, because the same networks measured weighted and unweighted give different answers and nothing else in the file tells them apart.

## GUI and CLI

`gui_app.py` opens a window, `run.py` takes a config file. Both end in the same two lines, so they run identical pipeline, generator, quality gate and save code. The only difference is where the config comes from.

The GUI can read a whole folder, one dataset per prefix, and a folder may hold a mixture:

| Files | Format |
| --- | --- |
| `<prefix>_edgelist.csv` + `<prefix>_positions.csv` | a CSV pair |
| `<prefix>_adjacency.npy` + `<prefix>_positions.npy` | a NumPy pair, positions transposed on load |
| `<prefix>_network.graphml` or `.graphml.gz` | one file, positions inside it |
| `<prefix>_image.tif` | the background for that prefix, optional |

A half named dataset stops the run instead of being skipped: an edge list with no positions, an adjacency with no coordinates, or one prefix claimed by two formats. Datasets are read in name order and share one run root and one manifest.

Two things this does not cover. The npy files in `data/input/samples/` keep older `_mat.npy` and `_pos.npy` names that CLI configs load directly, so those folders are not discoverable this way. And `analyse.py` reads a directory by its own rule, every GraphML file or else every CSV pair.

The form is a deliberate subset of what a config can express. Only the CLI can run `mosaic`, `scaling` and `compare`, set per dataset factors, fix tile frame sizes, ask for log spaced snapshots, or write SVG from a hybrid run. Going the other way, `INPUT_ORIENTATION` rotates the input as it is read and exists only in the form.

## Driving a run from another program

`gui_run.py` is the third entry point: one run from one JSON file, with no config module and nothing imported. It is how a host application uses NetworkSynth without becoming coupled to it.

```bash
python gui_run.py path/to/run_spec.json
```

```json
{
  "contract": 2,
  "inputs":  {"edge_list": "...", "positions": "...", "image": null},
  "output_dir": "/path/to/run_output",
  "mode": "generate",
  "params": {
    "SYNTHETIC_FRAME_SIZE": [1030, 730],
    "CLOSED_NODES_FACTOR": 1.2,
    "CLOSED_EDGES_FACTOR": 0.8,
    "SYNTHETIC_NETWORK_NUMBER": 10,
    "ERROR_TOLERANCE": 0.15,
    "ERROR_CHECKER": "multifractal",
    "MEASURE_WEIGHTED": false,
    "SEED": 12345
  }
}
```

The input shape depends on the mode, which is why `contract` is 2. `MODE_INPUTS` in `configs/gui_config.py` is the authority, and a spec must fill exactly one input set. Filling none is rejected, and so is filling two, because that does not say which input to read.

| Input set | Modes |
| --- | --- |
| `edge_list` + `positions`, `image` optional | `generate`, `hybrid`, `sweep` |
| `datasets_dir`, read by the prefix rule above | `generate`, `hybrid`, `sweep` |
| `original_dir` + `synthetic_dir` | `compare` |

A `sweep` spec adds `NF_RANGE` and `EF_RANGE` to `params`, and needs a wandb key in `WANDB_API_KEY` or `WANDB_KEY` or in `~/.netrc`. Login happens inside the run, so a missing key ends as a failed manifest with the reason rather than a hang. Start the process with stdin closed so wandb can never sit waiting for a key nobody can type.

Exit codes are `0` completed, `1` failed, `2` run spec rejected, `130` cancelled.

Every mode writes `manifest.json` at the run root, from a `finally`, so a failed or cancelled run still leaves one saying so:

```json
{
  "manifest_version": 1,
  "status": "ok",
  "run_root": "/path/to/output_dir/gui_generate_results_20260805_120000_abc123",
  "outputs": {
    "edge_lists": ["sample/synthetic/net_edgelist.csv"],
    "positions":  ["sample/synthetic/net_positions.csv"],
    "previews":   ["sample/synthetic/graph.webp"]
  },
  "file_count": 12
}
```

- Only kinds that have files appear, so read `outputs` with a default rather than by subscript. The kinds are `edge_lists`, `positions`, `networks`, `previews`, `snapshots`, `analysis` and `other`.
- `status` is `ok`, `failed` with an `error` string, or `cancelled`. Cancelled is deliberately separate, because a GUI must not show an error when the user pressed Cancel.
- A missing manifest means the process died hard.
- Paths are relative to `run_root`, with forward slashes on every platform.
- `manifest_version` and the spec's `contract` exist so a future change fails loudly instead of being misread.

Progress rides on the log: `{"tag": "PROGRESS", "percent": 66.67, ...}` in `run.jsonl`, next to `manifest.json`. Read the number rather than parsing prose.

Nothing in the spec records what an edge weight means. A `Weight` column may hold a diameter, area, length, angle, conductance or resistance depending on what produced it, and none of that is carried. The Mapper learns the length to weight relationship from the input and reproduces it, whatever the weight physically is. If the meaning ever stops being length related, that is a new mapper, not a new contract field.

## Troubleshooting

**Missing input files.** Check the paths match `BASE_INPUT_PATH` and the dataset IDs.

**Dimension mismatch.** Positions must be `(N, 2)` and the adjacency matrix `(N, N)`.

**Generation keeps failing.** Raise `CLOSED_NODES_FACTOR` to allow more merging, raise `ERROR_TOLERANCE` to accept less similar networks, or raise `MAX_ATTEMPTS`.

**A config will not load.** Each file needs exactly one class with `DATASETS`, found by that attribute rather than by name. The class must set `MODE`, or no pipeline claims it. And `SNAPSHOT_INTERVAL` needs saving on, since snapshots bypass the `Saver`, so pairing it with `DISABLE_SAVING` raises.

## The `dist` branch

`dist` is a code only copy of this repository, published so another project can carry NetworkSynth as a git submodule without the sample data tracked on `main`. It is an orphan branch holding just the code, `requirements.txt`, `LICENSE` and a README of its own.

Nothing is edited on `dist` directly, every commit there is generated:

```bash
scripts/publish_dist.sh [source-ref] [tag]     # defaults to HEAD, no tag
```

The script reads the allowlisted paths out of the source commit through a temporary index. The working tree is never read, so nothing untracked or ignored can reach the branch, and a publish works from a dirty branch or an old tag. It stops if an allowlisted path is missing, does nothing when the tree has not changed, and never pushes.

Only the first publish is an orphan commit. Later ones are its children, which is what lets a shallow submodule fetch work without force pushing.

`README_dist.md` in this repository is what ships as `README.md` on the branch. Both live here on purpose, since a file edited on `dist` would be overwritten by the next publish.

### Releasing

Tag `main` and push the tag. That is the whole release:

```bash
git tag -a vX.Y.Z -m "NetworkSynth vX.Y.Z"
git push origin vX.Y.Z
```

`.github/workflows/publish-dist.yml` picks it up, rebuilds `dist` from that commit, and tags it `dist-vX.Y.Z`. The dist tag is always derived from yours, so the two cannot drift apart. A tag that is not on `main` fails the job instead of publishing.

To publish by hand:

```bash
scripts/publish_dist.sh HEAD    # prints the push command
git push origin dist
```

### Taking a release, on the consumer side

A submodule records a commit, so a project carrying NetworkSynth keeps building against what it was pinned to until someone moves the pin. To take a named release:

```bash
git -C networksynth fetch --tags
git -C networksynth checkout dist-vX.Y.Z
```

To follow the tip of `dist` instead:

```bash
git -C networksynth fetch origin dist --depth 1
git -C networksynth checkout -B dist FETCH_HEAD
```

`git submodule update --remote` looks like the command for this but does nothing useful, because a submodule clone only tracks our default branch and never has an `origin/dist` to resolve.

Either way, the consuming project commits the gitlink afterwards with `git add networksynth`, or the new version is pinned only for whoever ran the commands.
