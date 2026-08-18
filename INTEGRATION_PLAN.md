# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.

Model: **loose coupling.** A button in their GUI opens NetworkSynth as a separate process with its own environment. They never import our code.

Status: **everything on our side is built and verified end to end on real sample data.** Every pipeline has been run for real, not just tested: the five modes the GUI offers each complete as a subprocess with the quality gate on and write a manifest, and `sweep` completes through its wandb agent. Worker processes use `spawn` everywhere, so nothing depends on the parent process surviving. What remains is a conversation with their maintainer about the contract in section 2, the work on their side in section 3, and the two open items in section 4. Completed work is not recorded here; git history holds it.

---

## 1. Why loose coupling makes this easy

Deciding on a separate process rather than an in-process library removes most of the hard problems before they start:

| Problem in an in-process merge | Under loose coupling |
|---|---|
| StructuralGT is on Python 3.14, networkit has no 3.14 wheels | **Gone.** We run in our own 3.12 environment. See Appendix A. |
| We spawn our own process pools; they own the Qt worker lifecycle | **Fine.** No contention. |
| `pipelines/generate.py` sets `OMP_NUM_THREADS` at import | **Fine.** Our process. |
| Need to vendor / subtree / publish a shared core package | **Not needed.** |
| Need an in-process `nx.Graph` ↔ `SynthGraph` adapter | **Not needed.** Handoff is files. |
| networkit vs igraph port (~2–3 days) | **Not needed. Keep networkit.** |

What is left is small: a file contract between the two, a launcher on their side, and a way to report progress back.

---

## 2. The file contract

**Inputs are chosen by the user, not handed over by StructuralGT.** StructuralGT decides its own export format and location; we let the user pick what to load.

Two input shapes:

| Shape | Files | Status |
|---|---|---|
| An explicit pair | an edge-list CSV and a positions CSV, named directly | **built** |
| A directory | every dataset in it, by naming convention | **not built** — see section 4 |

`graphs/csv_io.py` is deliberately tolerant about columns, so either origin works: it accepts StructuralGT's `Source,Target` (plus `Weight,Length,Width,Angle` when weighted) and our own `source_index,target_index,edge_weight`, with `x,y` for positions.

The directory convention already exists implicitly, because it is what we *write*: `{name}_edgelist.csv` and `{name}_positions.csv`. Anything we produce can therefore be re-loaded as input with no conversion. The `.npy` sample configs use a parallel convention, `{name}_mat.npy` + `{name}_pos.npy` + optional `{name}_image.tif`.

`run_spec.json` carries everything else:

```json
{
  "contract": 1,
  "inputs":  {"edge_list": "...", "positions": "...", "image": null},
  "output_dir": "/path/to/run_output",
  "mode": "generate",
  "params": {
    "FRAME_SIZE": [1030, 730],
    "SYNTHETIC_FRAME_SIZE": [1030, 730],
    "CLOSED_NODES_FACTOR": 1.2,
    "CLOSED_EDGES_FACTOR": 0.8,
    "SYNTHETIC_NETWORK_NUMBER": 10,
    "SYNTHETIC_GRAPH_NUMBER": 3,
    "ERROR_TOLERANCE": 0.15,
    "MEASURE_WEIGHTED": false,
    "SEED": 12345
  }
}
```

`SEED` matters: with it set, running twice with the same settings produces a byte-identical network. Workers get `SEED + worker_index`, so candidates still differ from each other while the run as a whole repeats.

**No field records what the weight means.** StructuralGT's `Weight` column can hold a diameter, area, length, angle, conductance or resistance depending on their setting, and we carry none of that: the Mapper learns the empirical length↔weight relationship from the input and reproduces it, whatever the weight physically is.

### What we write back

Every mode writes `manifest.json` at the run root, so their GUI never parses our directory names. It is written from a `finally`, so a run that fails or is cancelled still produces one saying so:

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

- `status` is `ok`, `failed` (with an `error` string) or `cancelled`. **`cancelled` is deliberately distinct from `failed`** — a GUI must not show an error because the user pressed Cancel.
- A *missing* manifest therefore means the process died hard, which is unambiguous rather than looking like a failed run.
- `snapshots` is separate from `previews`: the former is an animation sequence, the latter a final render.
- Paths are relative to `run_root` with forward slashes on every platform.
- `manifest_version` and the run-spec's `contract` exist so a future shape change fails loudly instead of being misread.

Because we hand back edge lists and positions in the same CSV shape they already read (`csv_to_graph`, `csv_to_numpy` in their `sgt_utils`), a generated network can re-enter StructuralGT through the existing `BaseController.add_graph()` path, and their whole analysis and PDF pipeline works untouched.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | completed |
| `1` | failed |
| `2` | the run-spec was rejected |
| `130` | cancelled (SIGINT) |

Worth knowing why the handler lives in the entry point rather than in a pipeline: `fire` swallows `SystemExit` and reports `2`, so a pipeline cannot set its own exit status. Pipelines log and re-raise; the entry point maps that to a code.

### Progress

Our JSON-lines log carries a numeric `percent` field on progress records (`{"tag": "PROGRESS", "percent": 66.67, …}`), so their controller can drive `ProgressData` by reading a field rather than parsing percentages out of prose. `run.jsonl` is written inside the run directory, next to `manifest.json`, at the path the run-spec named.

---

## 3. Their side: how deep to integrate

There are two depths, and this is the main thing to settle with their maintainer. The input model in section 2 makes the thin option genuinely viable.

**Thin — the button just opens our window.** The user picks input files in our GUI; our window writes its own run-spec and launches `gui_run.py`. Their side needs a ribbon button, an ini entry naming our interpreter and repo, and nothing else. No exporting, no spec writing, no progress plumbing, no manifest parsing. Results return whenever the user loads them through their existing `add_graph()` path.

| File | Change |
|---|---|
| `apps/qml/layouts/RibbonLayout.qml` | button after "Extract graph" (~line 257) |
| `sgt_configs.ini` | new `[synthesis-settings]` section (interpreter path, repo path) |

**Deep — their controller drives us headless.** Their side writes the run-spec, launches `gui_run.py`, tails `run.jsonl` for progress, reads `manifest.json`, and loads results automatically. More work for them, but the synthesis becomes part of their workflow rather than a side trip.

| File | Change |
|---|---|
| `apps/qml/layouts/RibbonLayout.qml` | button after "Extract graph" (~line 257) |
| `apps/qml/windows/SynthesisWindow.qml` | new: a `Window`, following `ImageHistogramWindow.qml` — mode selector, parameter form, Run, progress |
| `apps/qml/MainWindow.qml` | declare the window once (~line 83, beside `ImageHistogramWindow`) |
| `apps/controllers/synthesis_controller.py` | new: write run-spec, launch subprocess, stream progress, load results |
| `sgt_configs.ini` | new `[synthesis-settings]` section (interpreter path, default params) |
| `utils/config_loader.py` | `load_synth_configs()` returning the usual option-dict shape so QML binds for free |

Either way the launch is `<python> gui_run.py <path/to/run_spec.json>`, and `gui/app.py` on our side is a working reference implementation of the deep path — it writes a spec, launches the subprocess, and tails the log exactly as their controller would.

**Discovery and failure.** The ini file records where our interpreter and repo live. If either is missing, disable the button with a clear message rather than failing at click time. Non-zero exit → surface our stderr tail in their log window.

**If the deep path is chosen**, mirror their three memory layers so ours behaves like theirs:

| Layer | Theirs | Ours |
|---|---|---|
| Defaults | `sgt_configs.ini`, commented with valid ranges | `[synthesis-<mode>]` sections in the same ini |
| Runtime | option dicts `{id, type, text, value, minValue, maxValue, stepSize}` | `load_synth_configs()` returning the identical shape, so QML binds and widgets get their ranges for free |
| Session | `.sgtproj` (pickled state) | synthesis params ride along in it |
| Run record | — | `run_spec.json` written beside every run's output |

That last row is the one they do not have and we need: it is what makes a GUI run replayable, shareable, and re-runnable from the CLI.

The form should be **per-mode, not per-config-file**: configs within a mode differ only in values and paths, so the window offers a mode selector plus the handful of fields that vary, grouped by the same option-dict `type` field their `FiltersWidget.qml` already uses.

---

## 4. Open on our side

1. **Three of the eight pipelines are not reachable from the GUI, because they take different input.** The window offers `generate`, `generate_select`, `mosaic`, `scaling` and `hybrid` — all of which read an edge-list plus a positions CSV, which is why one fixed input form serves them. The others do not fit it:

   | mode | input it needs |
   | --- | --- |
   | `from_props` | a directory holding a `*_property.pkl` |
   | `analyze` | a directory holding `synthetic/` and `original/` |
   | `sweep` | a CSV pair, plus wandb credentials |

   So the work is not another row in a table — the window's **input section has to vary by mode**: a file pair for some, a directory for others. That means input descriptions alongside the parameter fields, per-mode validation (a directory containing a property pickle is a different check from two files existing), and an `inputs` shape in the run-spec that depends on the mode, which bumps `contract` to 2.

   The same change delivers **directory input** for the modes that already work — scanning a directory, grouping files by prefix, and turning each prefix into a dataset, which `DATASETS` already models. Two decisions come with that: what happens when a file is missing its partner (name the orphan and stop, matching the fail-fast stance elsewhere), and whether a directory of N datasets is one run producing N outputs or N runs. `RunPaths` and the manifest already support per-dataset subdirectories, so one run is the cheaper fit.

   `sweep` is deliberately excluded regardless. It hands control to wandb's agent to drive its own parameter search, so "Run" would mean "start a sweep and let wandb take over" — a different interaction from every other mode — and it needs credentials a GUI has no way to prompt for. It stays a CLI tool.

2. **Our GUI is unpolished.** It selects a mode, runs a generation and follows its progress, but has rough edges from local use that have not been worked through.

---

## 5. Sequencing

| Phase | Work | Verify | Owner |
|---|---|---|---|
| 0 | Agree the file contract and the integration depth (section 3) | written contract; no code | **their maintainer** — gates everything below |
| 1 | Per-mode input shapes: directory input, and `from_props` / `analyze` in the window | a directory of N datasets produces N outputs in one run; every `run.py` mode except `sweep` is reachable from the GUI | ours |
| 2 | Their launcher — thin or deep per phase 0 | click opens our window, or their controller runs us end to end | theirs |
| 3 | Results re-enter via `add_graph()` | a generated network opens as a new `sgt_obj` and their GT PDF works on it | theirs |

Phase 0 is the bottleneck. Nothing technical blocks it, and phase 1 is worth settling first because it changes the contract phase 0 would agree.

---

## 6. What we are deliberately not doing

- **No igraph port.** Keep networkit. It only mattered for an in-process merge.
- **No shared core package, subtree, or vendoring.** Nothing to keep in sync beyond the file contract in section 2 — which is the whole point of the loose model. Version the contract so a future change fails loudly.
- **No porting the pipelines.** They stay ours; the GUI selects among them by name in the run-spec's `mode` field.
- **No `sweep` in the GUI.** See section 4.
- **No carrying the meaning of edge weights.** See section 2.

---

## Appendix A: networkit and Python 3.14 — the facts

Checked 2026-07-31.

- Python 3.14 released **2025-10-07**.
- networkit shipped **11.2** (2025-11-05) and **11.2.1** (2026-01-09) after that date. Both ship wheels for cp310–cp313 only. No cp314, and no stable-ABI wheel.
- [Issue #1409 "Python 3.14 support"](https://github.com/networkit/networkit/issues/1409), opened 2026-05-08 by an outside user: still open, no assignee, no label, no milestone, no maintainer response, no stated timeline.
- Their docs still state "Python 3.13 and Numpy 2.X is fully supported."
- The project is actively maintained (wheel-upload automation merged 2026-01-13).
- Historical lag behind a new CPython: cp312 arrived ~4 months after Python 3.12; cp313 ~5 months after Python 3.13. We are now ~10 months past 3.14.

**Important nuance:** `pip install networkit` **does work** on Python 3.14 — it builds from the sdist. Only `uv` fails, which is what #1409 is really about. So this is a source build, not a wall. That is acceptable for a developer machine and unacceptable for a bundled end-user installer — which is another reason the loose model is the right call.
