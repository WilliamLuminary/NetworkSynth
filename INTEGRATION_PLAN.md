# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.

Model: **loose coupling.** A button in their GUI opens NetworkSynth as a separate process with its own environment. They never import our code.

Status: **everything on our side is built, tested, and verified end to end on real sample data.** What remains is a conversation with their maintainer about the contract in section 3, the work on their side in section 4, and two open design questions in section 5. Completed work has been removed from this document; git history holds the record.

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
| networkit vs igraph port (~2–3 days) | **Not needed. Keep networkit.** See Appendix B. |

What is left is small: a file contract between the two, a launcher on their side, and a way to report progress back.

---

## 2. How the bridge works on our side

Our configs are classes that already expose function slots meant to be overridden (`ORIGINAL_NETWORK_FUNC`, `ORIGINAL_IMAGE_FUNC`, `NETWORKS_FUNC`, `ATTRIBUTES_DICT_FUNC`). A GUI-driven run is just another config that overrides the loaders.

`configs/gui_config.py` sits at the `configs/` root rather than inside a `*_mode` package: it is not a mode, it is mode-*parameterised*, resolved at run time. `GuiConfig.from_spec(path)` validates a JSON run-spec and returns a **fresh subclass**, so two specs in one process cannot collide. `gui_run.py` is a dedicated entry point, which keeps `run.py` and the pipelines untouched.

Two details that will bite anyone extending this:

- `configs/_loader.py` globs `config_*.py` and returns only the **first** config class per module, so a single file holding several mode classes will not auto-register them all. The GUI entry point uses explicit dispatch instead.
- `BaseConfig.initialize()` derives `OUTPUT_DENOTE` from a module path segment ending in `_mode`. A config at the `configs/` root has none, so `GuiConfig` sets it explicitly.

There is no global mutable config state. Values reach worker processes as an immutable `SynthParams`, and the run's output location as a `RunPaths` value, so the pipelines work under a `spawn` start method — a GUI on Windows or macOS can drive them at all.

---

## 3. The file contract

**Inputs are chosen by the user, not handed over by StructuralGT.** StructuralGT decides its own export format and location; we let the user pick what to load. This is the correction that matters most in this document — an earlier draft had their exporter writing files for us to consume, which is not the model.

Two input shapes:

| Shape | Files | Status |
|---|---|---|
| An explicit pair | an edge-list CSV and a positions CSV, named directly | **built** |
| A directory | every dataset in it, by naming convention | **not built** — see section 5 |

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

**Deliberately not carried: what the weight *means*.** An earlier draft proposed a `weight_type` field, because StructuralGT's `Weight` column can hold a diameter, area, length, angle, conductance or resistance depending on their setting. Dropped, because nothing on our side would act on it: the Mapper learns the empirical length↔weight relationship from the input and reproduces it, whatever the weight physically is. If that relationship ever changes shape, the fix is a new Mapper, not a metadata field.

### What we write back

`handlers/manifest.py` writes `manifest.json` at the run root, so their GUI never parses our directory names:

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
- Written from a `finally`, so a *missing* manifest means the process died hard rather than being ambiguous with a failed run.
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

## 4. Their side: how deep to integrate

There are two depths, and this is the main thing to settle with their maintainer. The input model in section 3 makes the thin option genuinely viable, which was not obvious before.

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

## 5. Open on our side

1. **Directory input is not built.** `gui/spec_builder.py` takes one explicit pair of paths. Supporting a directory means scanning it, grouping files by prefix, and turning each prefix into a dataset — which `DATASETS` already models. Two decisions come with it: what happens when a file is missing its partner (name the orphan and stop, to match the fail-fast stance elsewhere), and whether a directory of N datasets is one run producing N outputs or N runs. `RunPaths` and the manifest already support per-dataset subdirectories, so one run is the cheaper fit. This also changes `inputs` in the run-spec, so `contract` bumps to 2.

2. **`read_graph_csv` has never seen a real StructuralGT export.** It is written against their documented column shape and tested only against CSVs we generate. One actual exported pair of files, run through it once, is the cheapest possible de-risking of the whole contract.

3. **Manifest covers `generate` and `generate_select` only.** `hybrid`, `hybrid_snapshot`, `mosaic`, `scaling` and `sweep` do not write one. Needed only for whichever modes the GUI ends up exposing, and the plan is to expose `generate` first.

4. **`spawn` versus `fork`.** Forked workers inherit an OpenMP runtime whose threads do not exist in them and spin instead of working; the fix applied was to pin each worker to one networkit thread, which `tests/test_worker_threads.py` now guards. Switching the start method to `spawn` would make that whole class of bug impossible, and the config refactor already made the pipelines spawn-safe. It is a behaviour change worth deciding on its own merits.

5. **Only `generate` has been verified by a real run.** `hybrid`, `mosaic`, `scaling` and `sweep` are exercised by the test suite but have not been run end to end on real data since the refactor. Given that a default-configuration hang survived a passing suite once, one real run each is worth the time before trusting them.

6. **Our GUI is unpolished.** It runs a generation and follows its progress, but has known rough edges from local use.

---

## 6. Sequencing

| Phase | Work | Verify | Owner |
|---|---|---|---|
| 0 | Agree the file contract and the integration depth (section 4) | written contract; no code | **their maintainer** — gates everything below |
| 1 | One real StructuralGT export through `read_graph_csv` | it loads, or the contract changes | ours, ~30 min |
| 2 | Directory input, if wanted | a directory of N datasets produces N outputs in one run | ours |
| 3 | Their launcher — thin or deep per phase 0 | click opens our window, or their controller runs us end to end | theirs |
| 4 | Results re-enter via `add_graph()` | a generated network opens as a new `sgt_obj` and their GT PDF works on it | theirs |

Phase 0 is the bottleneck. Nothing technical blocks it.

---

## 7. What we are deliberately not doing

- **No igraph port.** Keep networkit. It only mattered for an in-process merge. See Appendix B.
- **No shared core package, subtree, or vendoring.** Nothing to keep in sync beyond the file contract in section 3 — which is the whole point of the loose model. Version the contract so a future change fails loudly.
- **No porting the pipelines.** Hybrid, mosaic, sweep, scaling, snapshot and analyze stay ours. The GUI exposes `generate` first; others can be added to the run-spec's `mode` field later.
- **No carrying the meaning of edge weights.** See section 3.

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

---

## Appendix B: igraph vs networkit benchmark (kept for reference)

Run before the loose-coupling decision, when an in-process port was on the table. **No longer actionable**, but it documents that a port is viable if the integration model ever changes.

Measured on a 1821-node, 2591-edge weighted sample and synthetic sparse geometric graphs.

**Correctness:** shortest paths identical (`maxdiff = 0.00e+00`), betweenness identical (`corr = 1.000000`), clustering identical, components identical, unweighted diameter identical. Two genuine differences: networkit's weighted diameter returns an int and floors it (4.7 → 4, 0.3 → 0 — igraph is correct), and the two libraries define closeness differently for graphs in separate pieces (never arises, since `build_graph` always returns the largest component).

**Speed, networkit at 1 thread** (what all pipelines use, via `OMP_NUM_THREADS=1` and `nk.setNumberOfThreads(1)` in every pool worker): igraph was 1.3x faster on APSP at n=1821, 2.7–3.3x faster on betweenness, tied elsewhere, and 105x slower on exact diameter (avoidable — derive it from the APSP matrix).

**Speed, networkit at 16 threads** (only hybrid's phase 2 and `hybrid_snapshot`): igraph 3.0x slower on betweenness, 15.7x on closeness, 1.49x on APSP.

**Key finding:** the quality gate (`analyze_error_features`) depends only on `alpha_0` and `width`, both derived from APSP — which matched exactly. So a library swap could not change which candidates pass or fail. Everything that differs is used for reporting only.
