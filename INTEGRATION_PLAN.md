# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.

NetworkSynth runs as a separate process in its own Python 3.12 environment; StructuralGT never imports our code, and the two talk through files. Appendix A is the constraint that forces this.

**Both sides are built.** `generate`, `hybrid`, `sweep` and `compare` all run from our GUI as subprocesses, each writing a manifest; worker processes use `spawn`, so nothing depends on the parent surviving. StructuralGT's `gen` branch carries the button that opens us. What remains here is the part that is not visible from either repository's README: the file contract between us (section 1), how their side finds and launches ours (section 2), how our code reaches them (section 3), and the constraint that forces all of it (Appendix A).

---

## 1. What a run needs, and what it leaves behind

A run is one command: `<python> gui_run.py <path/to/run_spec.json>`. Everything else is in that file and in the directory it names.

### Inputs

Inputs are chosen by the user, not exported on our behalf, and what a mode reads depends on the mode. `MODE_INPUTS` in `configs/gui_config.py` is the authority: it lists the input *sets* each mode accepts, and a spec must fill exactly one. Filling none is rejected, and so is filling two — that does not say which input to read, and choosing silently would make a caller's mistake look like a working run.

| Input set | Files | Modes |
|---|---|---|
| `edge_list` + `positions` (+ optional `image`) | one network, named directly | `generate`, `hybrid`, `sweep` |
| `datasets_dir` | one dataset per prefix: `{name}_edgelist.csv` + `{name}_positions.csv`, or `{name}_adjacency.npy` + `{name}_positions.npy`, or `{name}_network.graphml` (`.graphml.gz` too) — each with an optional `{name}_image.tif` | `generate`, `hybrid`, `sweep` |
| `original_dir` + `synthetic_dir` | two directories of networks to compare, in either format we write — GraphML or CSV pairs | `compare` |

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

## 2. How StructuralGT launches us

One button on their ribbon opens our window. The user picks inputs in our GUI, which writes its own run-spec and launches `gui_run.py`, so nothing on their side exports files, writes specs, follows progress or reads manifests. Their `README.md` section 3(d) documents it for their users; `SynthesisController` in `src/sgtlib/apps/controllers/` is the whole of the code.

Four things about it are not obvious from either README, and are the reason this section exists.

**No fallback to their interpreter.** With nothing configured the button looks for us in `third_party/NetworkSynth` and for our interpreter in the `.venv` inside it; the `[synthesis-settings]` ini keys are overrides. It never falls back to `sys.executable`. Theirs cannot have networkit (Appendix A), and in the frozen build `sys.executable` is StructuralGT itself, so that fallback would relaunch their own application with `gui_app.py` as an argument.

**Qt plugin paths are stripped from our environment.** `QT_PLUGIN_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH` and the QML import paths point at their PySide6, or into their PyInstaller bundle. Without removing them our Qt loads theirs and the child dies on a plugin mismatch.

**Their GUI reads the packaged ini, not the repo-root one.** `MainController` is constructed with no config file, so `read_config_file` falls back to `sgtlib/utils/configs.ini` inside the package. The `sgt_configs.ini` at their repo root is the copy the terminal app takes with `-c`. Both carry the section, and both have to be kept in step.

**Failure is always visible before the click.** A missing checkout or environment disables the button and names the missing step in its tooltip; a non-zero exit puts the tail of our stderr in their SGT Logs window.

If the separate window ever becomes annoying enough to be worth replacing, `gui/app.py` is a working reference for driving us headless instead — it writes a spec, launches the subprocess, and tails `run.jsonl` exactly as one of their controllers would. Not planned.

---

## 3. How our code reaches them

They carry us as a git submodule at `third_party/NetworkSynth`, pinned to a commit on our `dist` branch — a generated, code-only branch of about 780 KB rather than the 162 MB a full clone costs. `.gitmodules` records `branch = dist`, `shallow = true` so the fetch is depth 1, and `update = none`.

`update = none` is there because we are a private repository inside a public one. Without it, anyone cloning StructuralGT with `--recurse-submodules` — a common habit — gets two `fatal:` lines and `Failed to clone 'sub' a second time, aborting`; the outer clone survives but looks broken. With it, git prints `Skipping submodule` and moves on. Cloning never touches us, no credentials are asked for, and the synthesis button simply stays disabled. Somebody who does have access opts in by name:

```bash
git submodule update --init --checkout third_party/NetworkSynth
```

`--checkout` is what overrides `update = none` — a plain `--init` silently skips — so it is the command their README gives and the one the button's own tooltip names. When we go public this can be relaxed, but nothing breaks if it never is.

Publishing is automatic: tagging a commit on `main` as `v1.0.1` triggers `.github/workflows/publish-dist.yml`, which rebuilds the branch from that commit and tags it `dist-v1.0.1`. The dist version is derived from the release tag, never chosen separately, so the two cannot drift. Our `README.md`, "The `dist` Branch", is the reference for both the branch and the workflow.

Moving the pin stays a deliberate commit on their side. That is the property worth protecting: a release of ours cannot change what their application runs, and any old checkout of their repository brings back the NetworkSynth it was built against.

---

## Appendix A: networkit and Python 3.14

Rechecked 2026-08-29. StructuralGT is on 3.14; networkit still publishes no cp314 wheel and no stable-ABI wheel, so it cannot be a dependency of their environment. **This is now close to resolving.**

- PyPI's latest is 11.2.1 (2026-01-09), cp310–cp313 only. No release of networkit has ever carried a cp314 wheel.
- [Issue #1409 "Python 3.14 support"](https://github.com/networkit/networkit/issues/1409) is still open with no assignee or milestone, but no longer unanswered. On 2026-08-14 the maintainer `fabratu` wrote that they built the 3.14 wheels and could not attach them, because PyPI no longer allows amending a release more than 14 days old — "We will create a new patch release shortly." Fifteen days on, that release has not appeared.
- A third party has published unofficial 3.14 wheels from a fork ([ggirelli/networkit 11.2.1-alpha](https://github.com/ggirelli/networkit/releases/tag/11.2.1-alpha-20260707-2)). Not something to depend on, but it does show the build works.
- Nuance: `pip install networkit` does work on 3.14 by building from the sdist; only `uv` fails, which is what #1409 is really about. A source build is acceptable on a developer machine and unacceptable in a bundled end-user installer.

When that patch release lands, the separate-process design stops being forced and becomes merely preferable — the two would still want separate environments, and the reasons in the last paragraph of this appendix do not depend on the wheel.

Separate processes sidestep it entirely, and also mean our process pools and the `OMP_NUM_THREADS` set at import in `pipelines/generate.py` never touch their Qt worker lifecycle.
