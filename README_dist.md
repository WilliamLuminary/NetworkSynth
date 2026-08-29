# NetworkSynth

Generates synthetic networks modelled on a real one: it measures an input network's structure, grows candidates that match it, and keeps the ones that pass a quality gate.

This is the **`dist` branch** — a generated, code-only copy of [NetworkSynth](https://github.com/WilliamLuminary/NetworkSynth) meant to be checked out as a submodule inside another project. It carries no test suite, no sample data, no notebooks and no development history. Nothing here should be edited: every commit on this branch is built from the main repository by `scripts/publish_dist.sh`, so a change made here is lost at the next publish. Fix things in the main repository instead.

## Setup

NetworkSynth needs its own Python environment, separate from whatever hosts it, because NetworKit publishes no wheel for Python 3.14.

```bash
python3.12 -m venv .venv          # 3.11-3.13 also work
.venv/bin/pip install -r requirements.txt
```

## Running it

The window, where inputs, parameters and the output folder are all chosen:

```bash
.venv/bin/python gui_app.py
```

The command line, where a config module holds the same settings:

```bash
.venv/bin/python run.py configs/generate_mode/config_sample.py
```

The config's `MODE` picks the pipeline — `generate`, `hybrid`, `mosaic`, `scaling`, `sweep` or `compare`. `run.py` with no argument lists every config shipped here. The sample configs read from `data/input/`, which this branch does not carry, so point `DATASETS` and `BASE_INPUT_PATH` at your own networks before running one.

A third entry point runs a single job from a JSON run-spec rather than a config module, which is how a host application drives NetworkSynth without importing it:

```bash
.venv/bin/python gui_run.py path/to/run_spec.json
```

It exits `0` completed, `1` failed, `2` run-spec rejected, `130` cancelled, and writes `manifest.json` and `run.jsonl` into the run directory whatever happens — `manifest.json` comes from a `finally`, so a failed or cancelled run still leaves one saying so.

## Inputs

A network is a pair of files sharing a prefix, or a single pickle:

| Files | Format |
| --- | --- |
| `<prefix>_edgelist.csv` + `<prefix>_positions.csv` | a CSV pair |
| `<prefix>_adjacency.npy` + `<prefix>_positions.npy` | a NumPy pair |
| `<prefix>_network.pkl` | a pickled network |
| `<prefix>_image.tif` | the background for that prefix, optional |

Point a run at a folder of these and it reads one dataset per prefix, in name order, writing a subdirectory each under one run root. A half-named dataset stops the run and names it rather than being passed over.

Edge lists are read tolerantly: `Source,Target` columns (with `Weight,Length,Width,Angle` when weighted) and `source_index,target_index,edge_weight` both work, with `x,y` for positions.

## Output

One directory per run, `{MODE}_{ConfigClass}_results_{timestamp}_{run_id}`, holding the original and synthetic networks as edge-list pairs, pickles and rendered images, plus a per-dataset `report.txt`. `manifest.json` at the run root groups every file written by kind and records whether the run ended `ok`, `failed` or `cancelled`.

## Full documentation

Everything else — the configuration system, the save and render layers, hybrid tiling, parameter sweeps, log analysis — is in the [main repository](https://github.com/WilliamLuminary/NetworkSynth).
