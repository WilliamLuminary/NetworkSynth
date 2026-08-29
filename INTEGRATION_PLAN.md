# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.

NetworkSynth runs as a separate process in its own Python 3.12 environment; StructuralGT never imports our code, and the two talk through files. Appendix A is the constraint that forces this.

**NetworkSynth is done.** `generate`, `hybrid`, `sweep` and `compare` all run from our GUI as subprocesses, each writing a manifest, and every CLI mode runs for real on sample data. Worker processes use `spawn`, so nothing depends on the parent surviving. **The StructuralGT side is done too**: section 2 lists what changed on their `gen` branch.

---

## 1. What a run needs, and what it leaves behind

A run is one command: `<python> gui_run.py <path/to/run_spec.json>`. Everything else is in that file and in the directory it names.

### Inputs

Inputs are chosen by the user, not exported on our behalf, and what a mode reads depends on the mode. `MODE_INPUTS` in `configs/gui_config.py` is the authority: it lists the input *sets* each mode accepts, and a spec must fill exactly one. Filling none is rejected, and so is filling two — that does not say which input to read, and choosing silently would make a caller's mistake look like a working run.

| Input set | Files | Modes |
|---|---|---|
| `edge_list` + `positions` (+ optional `image`) | one network, named directly | `generate`, `hybrid`, `sweep` |
| `datasets_dir` | one dataset per prefix: `{name}_edgelist.csv` + `{name}_positions.csv`, or `{name}_adjacency.npy` + `{name}_positions.npy`, or `{name}_network.pkl` — each with an optional `{name}_image.tif` | `generate`, `hybrid`, `sweep` |
| `original_dir` + `synthetic_dir` | two directories of networks to compare, in either format we write — a pickled batch or CSV pairs | `compare` |

Comparison stands alone: `compare` reads the two sets of networks the caller names, and generation never analyses what it just made. So "generate then compare" is two runs — the second pointed at the first's `original/` and `synthetic/` folders, or at any other pair. Generation's own quality gate is a separate, cheaper thing: an `ErrorChecker` chosen by `ERROR_CHECKER`, measuring two scalars rather than a full spectrum.

`graphs/csv_io.py` is deliberately tolerant about columns, so either origin works: StructuralGT's `Source,Target` (plus `Weight,Length,Width,Angle` when weighted) or our own `source_index,target_index,edge_weight`, with `x,y` for positions.

A `datasets_dir` becomes one dataset per prefix in it, in name order, so one run writes N subdirectories under a single run root with one manifest covering them all. The files a prefix carries decide its format, so a directory may hold a mixture of the three. A half-named dataset stops the run and names it: an edge list with no positions beside it, an adjacency with no coordinates, or one prefix named as two datasets at once.

### The run-spec

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

`contract` is 2 because the input shape varies by mode. A `compare` spec carries `"inputs": {"original_dir": "...", "synthetic_dir": "..."}`, a directory run carries `"inputs": {"datasets_dir": "..."}`, and a `sweep` spec adds `NF_RANGE` and `EF_RANGE` to `params`.

With `SEED` set, the same settings reproduce the same networks byte for byte; workers get `SEED + worker_index`, so candidates still differ from each other while the run as a whole repeats.

No field records what an edge weight *means*. StructuralGT's `Weight` column can hold a diameter, area, length, angle, conductance or resistance depending on their setting, and we carry none of it: the Mapper learns the empirical length↔weight relationship from the input and reproduces it, whatever the weight physically is. If the meaning ever stops being length-related, that is a new mapper, not a new contract field.

### The manifest

Every mode writes `manifest.json` at the run root, from a `finally`, so a run that fails or is cancelled still produces one saying so:

```json
{
  "manifest_version": 1,
  "status": "ok",
  "run_root": "/path/to/output_dir/gui_generate_results_20260805_120000_abc123",
  "outputs": {
    "edge_lists": ["sample/synthetic/net_edgelist.csv"],
    "positions":  ["sample/synthetic/net_positions.csv"],
    "previews":   ["sample/synthetic/graph.webp"],
    "snapshots":  [],
    "networks":   [],
    "analysis":   []
  },
  "file_count": 12
}
```

- `status` is `ok`, `failed` (with an `error` string) or `cancelled`. `cancelled` is deliberately distinct from `failed` — a GUI must not show an error because the user pressed Cancel.
- A *missing* manifest means the process died hard, which is unambiguous rather than looking like a failed run.
- `snapshots` is an animation sequence; `previews` is a final render; `analysis` is what `compare` produces, kept apart from both.
- Paths are relative to `run_root`, forward slashes on every platform.
- `manifest_version` and the spec's `contract` exist so a future shape change fails loudly instead of being misread.

### Exit codes and progress

`0` completed, `1` failed, `2` run-spec rejected, `130` cancelled. The handler lives in the entry point rather than in a pipeline because `fire` swallows `SystemExit` and reports `2` — pipelines log and re-raise, and the entry point maps that to a code.

Progress rides on our JSON-lines log: `{"tag": "PROGRESS", "percent": 66.67, …}` in `run.jsonl`, written inside the run directory next to `manifest.json`. A reader drives a progress bar off a numeric field rather than parsing prose.

A sweep additionally needs a wandb key, in `WANDB_API_KEY` or `WANDB_KEY` (both are read), or in `~/.netrc`. Login happens inside the run's try block, so a missing key ends up in the manifest as `failed` with the reason, and the subprocess runs with stdin closed so wandb can never sit waiting for a key nobody can type.

---

## 2. The StructuralGT side

Done, on branch `gen`. One button on their ribbon opens our window. The user picks their inputs in our GUI, which writes its own run-spec and launches `gui_run.py`, so nothing on their side has to export files, write specs, follow progress, or read manifests.

| File | Change |
|---|---|
| `src/sgtlib/apps/controllers/synthesis_controller.py` | new: reads the settings, says why the button is disabled, starts our `gui_app.py` with `QProcess` |
| `src/sgtlib/apps/controllers/main_controller.py` | holds it as `synth_ctrl` |
| `src/sgtlib/apps/gui_app.py` | exposes it to QML as `synthesisController` |
| `src/sgtlib/apps/qml/layouts/RibbonLayout.qml` | button after "Extract graph" |
| `src/sgtlib/apps/qml/assets/icons/synth_icon.png` | new: node-link glyph with a plus, in the ribbon's line-art style |
| `src/sgtlib/utils/config_loader.py` | `load_synthesis_configs` |
| `src/sgtlib/utils/configs.ini`, `sgt_configs.ini` | new `[synthesis-settings]`: `python_interpreter`, `repo_dir` |
| `README.md` | section 3(d) |

Neither setting has to be filled in. With nothing configured the button looks for us in `third_party/NetworkSynth` — where a submodule checkout lands — and for our interpreter in the `.venv` inside it, on either platform's layout; the two ini keys are overrides, each of which stands on its own. What it never does is fall back to their own interpreter: theirs cannot have networkit, and in the frozen build `sys.executable` is StructuralGT itself, so that fallback would relaunch their app rather than ours.

Until both the checkout and its environment are there the button is disabled and its tooltip names the missing step, down to the `git submodule update --init` that fetches us. Nothing fails at click time. A non-zero exit puts the tail of our stderr in their SGT Logs window.

Two details worth remembering. Our process is started without the Qt plugin-path variables StructuralGT sets for itself, so our Qt loads its own plugins rather than theirs — without that the child dies on a plugin mismatch in the frozen build. And their GUI reads its settings from the `configs.ini` inside the package, not from the `sgt_configs.ini` in their repo root: that one is the copy the terminal app takes with `-c`. Both carry the section.

If the separate window ever becomes annoying enough to be worth replacing, `gui/app.py` is a working reference for driving us headless instead — it writes a spec, launches the subprocess, and tails `run.jsonl` exactly as one of their controllers would. Not planned.

---

## Appendix A: networkit and Python 3.14

Checked 2026-07-31. StructuralGT is on 3.14; networkit publishes no cp314 wheel and no stable-ABI wheel, so it cannot be a dependency of their environment.

- networkit 11.2 (2025-11-05) and 11.2.1 (2026-01-09) both ship cp310–cp313 only; their docs still state 3.13 is the supported ceiling.
- [Issue #1409 "Python 3.14 support"](https://github.com/networkit/networkit/issues/1409), opened 2026-05-08 by an outside user: still open, no assignee, no milestone, no maintainer response.
- The project is actively maintained, so this will resolve eventually — historically cp312 landed ~4 months after Python 3.12 and cp313 ~5 months after 3.13. We are ~10 months past 3.14.
- Nuance: `pip install networkit` does work on 3.14 by building from the sdist; only `uv` fails, which is what #1409 is really about. A source build is acceptable on a developer machine and unacceptable in a bundled end-user installer.

Separate processes sidestep it entirely, and also mean our process pools and the `OMP_NUM_THREADS` set at import in `pipelines/generate.py` never touch their Qt worker lifecycle.
