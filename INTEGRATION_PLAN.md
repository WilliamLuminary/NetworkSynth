# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.
Model: **loose coupling.** A button in their GUI launches NetworkSynth as a
separate process with its own environment. They never import our code.
Status: **everything on our side is built and tested** — seeding, result
manifest, CSV reader, exit codes, structured progress, run logging, the config
refactor that makes the pipelines work under a `spawn` start method, the
`gui_run.py` entry point, and a working GUI of our own that drives it. Nothing
has been written against StructuralGT itself, and **the file contract in section
3 has not been agreed with their maintainer** — that conversation now gates the
work. See section 5 for what is done and what genuinely remains.

---

## 1. Why loose coupling makes this easy

Deciding on a separate process rather than an in-process library removes most of
the hard problems before they start:

| Problem in an in-process merge | Under loose coupling |
|---|---|
| StructuralGT is on Python 3.14, networkit has no 3.14 wheels | **Gone.** We run in our own 3.12 environment. |
| `BaseConfig` mutates class-level global state | **Fine.** Every run is a fresh process — and this has since been removed anyway. |
| `BaseConfig._setup_logger()` hijacks the root logger | **Gone.** Logging moved to the entry points; `run.jsonl` is written inside the run directory. |
| `pipelines/generate.py:12` sets `OMP_NUM_THREADS` at import | **Fine.** Our process. |
| We spawn our own process pools; they own the Qt worker lifecycle | **Fine.** No contention. |
| Need to vendor / subtree / publish a shared core package | **Not needed.** |
| Need an in-process `nx.Graph` ↔ `SynthGraph` adapter | **Not needed.** Handoff is files. |
| networkit vs igraph port (~2–3 days) | **Not needed. Keep networkit.** |

What's left is a genuinely small amount of work: a config subclass on our side,
a file contract between the two, a launcher button on theirs, and a way to
report progress back.

---

## 2. The bridge: use the config system as designed

Our configs are classes, not instances, and they already expose function slots
meant to be overridden (`ORIGINAL_NETWORK_FUNC`, `ORIGINAL_IMAGE_FUNC`,
`NETWORKS_FUNC`, `ATTRIBUTES_DICT_FUNC`). A GUI-driven run is just another
config that overrides the loaders to read what StructuralGT exported.

**`configs/gui_config.py`** — at the `configs/` root, not inside a `*_mode`
package: it is not a mode, it is mode-*parameterised*, resolved at run time.

```python
class GuiConfig(BaseConfig):
    MODE: str = "generate"

    @classmethod
    def from_spec(cls, spec_path):
        """Returns a *fresh subclass*, so two specs in one process cannot collide."""
        spec = _read_and_validate(spec_path)      # missing keys / bad contract raise
        run = type("GuiRunConfig", (cls,), {})
        run.MODE = spec["mode"]
        for key, value in spec["params"].items():
            setattr(run, key, value)          # FRAME_SIZE, factors, counts, SEED…
        run.DATASETS = [DatasetId(spec.get("run_name", "gui_run"))]
        run.BASE_OUTPUT_PATH = spec["output_dir"]
        run._paths = spec["inputs"]
        return run

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.OUTPUT_DENOTE = f"gui_{cls.MODE}"   # explicit: see note below
        for name, fn in _MODE_SLOTS[cls.MODE].items():
            setattr(cls, name, fn)

    @staticmethod
    def load_original_network(dataset_id):
        from graphs import read_graph_csv
        return read_graph_csv(
            GuiConfig._paths["edge_list"], GuiConfig._paths["positions"]
        )
```

Three things make this cheap:

- **One file covers every mode.** Under GUI control the input always arrives the
  same way, so the loader functions are identical for generate / hybrid / mosaic
  / scaling. Only `analyze` (`NETWORKS_FUNC`) and `attr_generate`
  (`ATTRIBUTES_DICT_FUNC`) differ, so `_MODE_SLOTS` is about three entries.
- **`read_graph_csv` already exists** (`graphs/csv_io.py`) and reads their export
  format directly. See section 3.
- Numbers arrive from a JSON run-spec rather than being hardcoded, which keeps
  the process boundary clean and makes every run inspectable and repeatable.

Two details that will bite otherwise:

- `configs/_loader.py` globs `config_*.py` and returns only the **first** config
  class it finds per module. So a single file holding several mode classes will
  not auto-register them all — use explicit dispatch in the GUI entry point
  rather than relying on discovery.
- `BaseConfig.initialize()` derives `OUTPUT_DENOTE` by looking for a module path
  segment ending in `_mode`. A config at the `configs/` root has none, hence
  setting it explicitly above.

**Config state is no longer global.** `_inject_dependencies()` has been deleted;
values reach worker processes as an immutable `SynthParams` passed explicitly,
and the run's output location as a `RunPaths` value. Two consequences for this
plan: the pipelines now work under a `spawn` start method (so a GUI on Windows or
macOS can drive them at all), and there is no longer any barrier to serving more
than one run from one interpreter.

## 3. The file contract

StructuralGT already has the export options we need — `load_gte_configs`
exposes `export_edge_list`, `export_node_positions`, and `export_adj_mat`. So
their side needs no new serialization code.

**They write** (into a temp run directory):

| File | Produced by | Read by us via |
|---|---|---|
| `edge_list.csv` | existing `export_edge_list` | `graphs.read_graph_csv` |
| `positions.csv` | existing `export_node_positions` | same |
| `image.tif` *(optional)* | existing image save | `load_original_image` |
| `run_spec.json` | new, small | `GuiConfig.from_spec` |

`run_spec.json`:

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

`SEED` matters: with it set, clicking Run twice with the same settings produces a
byte-identical network. Workers get `SEED + worker_index`, so candidates still
differ from each other while the run as a whole repeats.

**Deliberately not carried: what the weight *means*.** An earlier draft of this
plan proposed a `weight_type` field, because StructuralGT's `Weight` column can
hold a diameter, area, length, angle, conductance or resistance depending on
their setting. Dropped, because nothing on our side would act on it: the Mapper
learns the empirical length↔weight relationship from the input and reproduces
it, whatever the weight physically is. If that relationship ever changes shape,
the fix is a new Mapper, not a metadata field.

### What we write back

`handlers/manifest.py` writes `manifest.json` at the run root, so their GUI never
parses our directory names:

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

- `status` is `ok`, `failed` (with an `error` string) or `cancelled`.
  **`cancelled` is deliberately distinct from `failed`** — a GUI must not show an
  error because the user pressed Cancel.
- Written from a `finally`, so a *missing* manifest means the process died hard
  rather than being ambiguous with a failed run.
- `snapshots` is separate from `previews`: the former is an animation sequence,
  the latter a final render.
- Paths are relative to `run_root` with forward slashes on every platform.
- `manifest_version` and the run-spec's `contract` field exist so a future shape
  change fails loudly instead of being misread.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | completed |
| `1` | failed |
| `130` | cancelled (SIGINT) |

Worth knowing why the handler lives in `run.py` rather than in a pipeline:
`fire` swallows `SystemExit` and reports `2`, so a pipeline cannot set its own
exit status. Pipelines log and re-raise; the entry point maps that to 130.

### Progress

Our JSON-lines log carries a numeric `percent` field on progress records
(`{"tag": "PROGRESS", "percent": 66.67, …}`), so their controller can drive
`ProgressData` by reading a field rather than parsing percentages out of prose.

**Done.** `handlers/run_logging.py` attaches the run log inside the run
directory, so `run.jsonl` sits next to `manifest.json` at the path the run-spec
named. Their controller tails that file.

Because we hand back edge lists and positions in the same CSV shape they
already read (`csv_to_graph`, `csv_to_numpy` in their `sgt_utils`), a generated
network can re-enter StructuralGT through the existing
`BaseController.add_graph()` path and their whole analysis and PDF pipeline
works untouched.

---

## 4. Their side: the button

Minimal, and it touches little:

| File | Change |
|---|---|
| `apps/qml/layouts/RibbonLayout.qml` | button after "Extract graph" (~line 257) |
| `apps/qml/windows/SynthesisWindow.qml` | new: a `Window`, following `ImageHistogramWindow.qml` — mode selector, parameter form, Run, progress |
| `apps/qml/MainWindow.qml` | declare the window once (~line 83, beside `ImageHistogramWindow`) |
| `apps/controllers/synthesis_controller.py` | new: write run-spec, export CSVs, launch subprocess, stream progress, load results |
| `sgt_configs.ini` | new `[synthesis-settings]` section (interpreter path, default params) |
| `utils/config_loader.py` | `load_synth_configs()` returning the usual option-dict shape so QML binds for free |

The controller launches us with `subprocess.Popen`, pointed at the interpreter
named in the ini file:

```
<python> gui_run.py <path/to/run_spec.json>
```

A **dedicated entry point** rather than `-m run …`: it keeps `run.py` and the
pipelines untouched, and lets the GUI path resolve its mode from the run-spec
instead of squeezing that through CLI flags.

**Progress reporting.** Their controller tails our JSON-lines log and forwards
each line as a `ProgressData`. Progress records carry a numeric `percent` field,
so no parsing of message text — see section 3.

**Discovery and failure.** The ini file records where our interpreter and repo
live. If either is missing, disable the button with a clear message rather than
failing at click time. Non-zero exit → surface our stderr tail in their log
window.

---

## 5. Groundwork on our side

**Done.**

| | Delivered |
|---|---|
| Reproducible seeding | `SEED` in the config, threaded through; workers get `SEED + index`. Same seed twice → byte-identical network. |
| Result manifest | `handlers/manifest.py`; see section 3. Wired into `generate` and `generate_select`. |
| CSV reader | `graphs/csv_io.py` reads their export format directly. |
| Exit codes | 0 / 1 / 130; cancellation no longer reports success. |
| Structured progress | numeric `percent` in the JSON log. |
| Config refactor | no global mutable config state; the pipelines work under `spawn`, so a GUI on Windows or macOS can drive them at all. |
| Logging | `handlers/run_logging.py`; `run.jsonl` lands inside the run directory, one handler per run. |
| GUI entry path | `configs/gui_config.py` + `gui_run.py`: a hand-written `run_spec.json` produces a network and a manifest, no GUI involved. |
| Our own GUI | `gui/` + `gui_app.py` (PySide6 + QML, matching their toolkit). Launches `gui_run.py` as a subprocess and tails `run.jsonl` — the same mechanism their controller will use, so it doubles as the reference implementation. |
| Weighted analysis | previously crashed on every real input; now runs. See Appendix C.2. |

**Outstanding.** Items 1, 3 and 4 below are resolved and kept only as a record;
the genuinely open items are 2, 5 and 6.

1. **Logging.** Resolved. Moved out of `BaseConfig` into `handlers/run_logging.py`;
   the entry points call it, and `run.jsonl` is written inside the run directory.

2. **Manifest for the remaining pipelines.** `generate` and `generate_select`
   write one; hybrid, hybrid_snapshot, mosaic, scaling and sweep do not yet.
   Only needed for whichever modes the GUI ends up exposing.

3. **Zero-weight edges.** Resolved — see Appendix C.2. They came from the
   source pool the Mapper samples, a zero weight is now rejected outright, and
   the file named here has been replaced by
   `tests/data/A_10kX_weighted_network_positive.pkl`.

4. **Redundant APSP.** Resolved — see Appendix C.1. Note the "roughly halves
   analysis cost" claim here was wrong: measured properly it is worth about 1%.
   The reason it was kept is that curvature no longer duplicates the distance-graph
   construction.

5. **`weight_type`.** Dropped from the contract — see section 3. It was never
   implemented and nothing would have used it.

6. **`read_graph_csv` has never seen a real StructuralGT export.** It is written
   against their documented column shape and is tested against CSVs we generate
   ourselves. One actual exported pair of files, run through it once, is the
   cheapest possible de-risking of the whole file contract.

---

## 6. Sequencing

| Phase | Work | Verify | State |
|---|---|---|---|
| 0 | Agree the file contract and the button's scope with their maintainer | written contract; no code | **their call** |
| 1 | `SEED` + thread it through | same seed twice → identical output | **done** |
| 2 | Result manifest | manifest lists every produced file; paths resolve | **done** (generate, generate_select) |
| 3 | CSV reader | reads their export format | **done** |
| 4 | Exit codes + structured progress | 130 on cancel; numeric `percent` in the log | **done** |
| 5 | Move logging to the entry points | log lands inside the run-spec's `output_dir` | **done** |
| 6 | `configs/gui_config.py` + run-spec loader + `gui_run.py` | a hand-written `run_spec.json` produces a network from CSV inputs, no GUI involved | **done** |
| 6b | Our own GUI (`gui_app.py`) driving the same entry point | window runs a generation and follows its progress | **done**, unpolished |
| 7 | Their `synthesis_controller.py` + subprocess launch, headless | controller runs us end to end, parses the manifest, reads progress | todo |
| 8 | `SynthesisWindow.qml` + ribbon button + `[synthesis-settings]` ini section | click-through in the GUI; progress bar advances; Cancel terminates the child and yields `cancelled` | todo |
| 9 | Results re-enter via `add_graph()` | a generated network opens as a new `sgt_obj` and their GT PDF works on it | todo |

Phases 1–6 are entirely on our side and are **complete**. Phases 7–9 are theirs, and small. Phase 0 — agreeing the contract — has not happened and gates everything after it.

**Note on phase 8:** a `Window`, not a widget — following their existing
`ImageHistogramWindow.qml` pattern (a `Window { }` with `import Theme 1.0`,
declared once in `MainWindow.qml`, shown via `visible = true`). That inherits
their theme for free. The form should be **per-mode, not per-config-file**:
configs within a mode differ only in values and paths, so the window offers a
mode selector plus the handful of fields that vary, grouped using the same
option-dict `type` field their `FiltersWidget.qml` already groups by.

**Memory, mirroring their three layers** so ours behaves like theirs:

| Layer | Theirs | Ours |
|---|---|---|
| Defaults | `sgt_configs.ini`, commented with valid ranges | `[synthesis-<mode>]` sections in the same ini |
| Runtime | option dicts `{id, type, text, value, minValue, maxValue, stepSize}` | `load_synth_configs()` returning the identical shape, so QML binds and the widgets get their ranges for free |
| Session | `.sgtproj` (pickled state) | synthesis params ride along in it |
| Run record | — | `run_spec.json` written beside every run's output |

That last row is the one they do not have and we need: it is what makes a GUI run
replayable, shareable, and re-runnable from the CLI.

---

## 7. What we are deliberately not doing

- **No igraph port.** Keep networkit. It only mattered for an in-process merge.
- **No shared core package, subtree, or vendoring.** Nothing to keep in sync
  beyond the file contract in section 3 — which is the whole point of the loose
  model. Version the contract (`"contract": 1` in the run-spec) so a future
  change fails loudly.
- **No porting the pipelines.** Hybrid, mosaic, sweep, scaling, snapshot and
  analyze stay ours. The GUI exposes `generate` first; others can be added to
  the run-spec's `mode` field later if wanted.
- **No refactor of `BaseConfig`'s global state.** It is only a problem
  in-process, and we are not in-process.

---

## Appendix A: networkit and Python 3.14 — the facts

Checked 2026-07-31.

- Python 3.14 released **2025-10-07**.
- networkit shipped **11.2** (2025-11-05) and **11.2.1** (2026-01-09) after
  that date. Both ship wheels for cp310–cp313 only. No cp314, and no stable-ABI
  wheel.
- [Issue #1409 "Python 3.14 support"](https://github.com/networkit/networkit/issues/1409),
  opened 2026-05-08 by an outside user: still open, no assignee, no label, no
  milestone, no maintainer response, no stated timeline.
- Their docs still state "Python 3.13 and Numpy 2.X is fully supported."
- The project is actively maintained (wheel-upload automation merged 2026-01-13).
- Historical lag behind a new CPython: cp312 arrived ~4 months after Python
  3.12; cp313 ~5 months after Python 3.13. We are now ~10 months past 3.14.

**Important nuance:** `pip install networkit` **does work** on Python 3.14 — it
builds from the sdist. Only `uv` fails, which is what #1409 is really about. So
this is a source build, not a wall. That is acceptable for a developer machine
and unacceptable for a bundled end-user installer — which is another reason the
loose model is the right call.

---

## Appendix B: igraph vs networkit benchmark (kept for reference)

Run before the loose-coupling decision, when an in-process port was on the
table. **No longer actionable**, but it documents that a port is viable if the
integration model ever changes.

Measured on the then-current `A_10kX_weighted_network.pkl` (n=1821, e=2591; since
replaced, see Appendix C.2) and
synthetic sparse geometric graphs.

**Correctness:** shortest paths identical (`maxdiff = 0.00e+00`), betweenness
identical (`corr = 1.000000`), clustering identical, components identical,
unweighted diameter identical. Two genuine differences: networkit's weighted
diameter returns an int and floors it (4.7 → 4, 0.3 → 0 — igraph is correct),
and the two libraries define closeness differently for graphs in separate
pieces (never arises, since `build_graph` always returns the largest component).

**Speed, networkit at 1 thread** (what all pipelines actually use, via
`OMP_NUM_THREADS=1` and `nk.setNumberOfThreads(1)` at `hybrid.py:159,247`):
igraph was 1.3x faster on APSP at n=1821, 2.7–3.3x faster on betweenness, tied
elsewhere, and 105x slower on exact diameter (avoidable — derive it from the
APSP matrix).

**Speed, networkit at 16 threads** (only `hybrid.py:570` Phase 2 and
`hybrid_snapshot.py:184`): igraph 3.0x slower on betweenness, 15.7x on
closeness, 1.49x on APSP.

**Key finding:** the quality gate (`analyze_error_features`) depends only on
`alpha_0` and `width`, both derived from APSP — which matched exactly. So a
library swap could not change which candidates pass or fail. Everything that
differs is used for reporting only.

## Appendix C: measured issues in `analysis/multifractal_analyzer.py`

Found while benchmarking. C.1 and C.3 are **implemented**; C.2 is a recorded decision, deliberately left crashing. Timings were re-taken on an idle machine after an earlier round was contaminated by a concurrent test run.

### C.1 A full analysis runs APSP three times — implemented, but for code reasons not speed

`analyze_graph()` computed all-pairs shortest paths three separate times over the same distances: `_compute_multifractal_taus` and `_compute_node_dimension` on the analysis graph, and `_ollivier_ricci_curvature` on a private copy it built itself. Now memoised per distinct graph: 3 runs becomes 1 (unweighted) or 2 (weighted, which genuinely uses two graphs).

**Correcting an earlier claim in this document.** It previously said this was worth 48.5s → 26.0s, a 1.9x speedup. That was wrong — those timings were taken while a test suite ran concurrently, so the difference was load, not the cache. Measured properly, one APSP costs **0.03s at 705 nodes, 0.14s at 1821 nodes, 0.16s weighted (Dijkstra) at 1821**. Removing two of three saves ~0.1s out of a ~32s analysis: **about 1%.** The real cost of `analyze_graph` is elsewhere — chiefly the per-edge Wasserstein LP in the curvature loop.

Output is unchanged: every APSP-derived metric (`tau_list`, `dim_list`, `nfd_dist`, `ricci_dist`, `closeness_dist`, `diameter`) is bit-identical. `betweenness_dist` and `eigen_dist` move by ~1e-16, but they use no APSP and drift by the same amount when unmodified code is run twice — networkit's threaded nondeterminism.

**The reason to keep it is single-source-of-truth, not performance.** `_ollivier_ricci_curvature` was duplicating the graph-construction logic that `_get_analysis_graph` and `_get_inverted_weight_graph` already own. That duplication is live: C.2 below is an open decision about exactly how `_get_inverted_weight_graph` should treat zero weights, and under the old code any such change would silently *not* apply to curvature, which built its own graph with its own copy of the `1/w` rule. Now there is one construction and one distance matrix.

The saving does grow with graph size — APSP is O(V·E) — but so does the memory below, and faster.

### C.2 Weighted analysis could not run — fixed, and the original diagnosis was wrong

Weighted `analyze_graph()` aborted with `ValueError: Invalid input for linprog: b_eq must not contain values inf, nan, or None`.

**The first diagnosis in this document blamed zero-weight edges. That was only the most extreme case of a general numeric problem, and fixing the zeros did not fix the crash.**

Curvature spreads mass over neighbours in proportion to `base ** -(d ** exp_power)`, with `d = 1/w`. A double underflows to exactly `0.0` once `(1/w)**2 > ~745`, i.e. `w < 0.0366`. Real edge widths sit below that: the weighted fixture has median weight **0.0312**, with **1426 of 2506 edges** under the limit and **442 nodes where every neighbour underflows**. Those nodes normalise `0/0`, and the resulting NaN aborts the transport solver. Samples A–D have median weights of 0.028–0.048, so this affects real runs, not just fixtures.

The convention `base=e, exp_power=2` is inherited from GraphRicciCurvature, which assumes distances near 1. Inverted widths here are ~30, and `e ** -900` is zero.

Fixed in `_neighbour_masses` by computing the normalised affinities as a shifted softmax — subtracting the largest exponent before exponentiating. Numerator and denominator scale by the same constant, so it cancels exactly: the same number, kept inside the representable range. Nothing is clamped, floored or defaulted. Verified to match the naive expression to 1e-15 where that expression does not underflow.

Weighted `analyze_graph()` now completes in 8.0s with zero NaNs, producing 2506 curvature values in [-0.95, 1.00].

**Separately**, a zero weight is now rejected outright. `SynthGraph.set_weight` and `SynthGraph.from_sparse_matrix` assert `w > 0`, naming the offending edge — an edge that exists but has no width is not a measurable graph. The Mapper samples weights from the source network with replacement, so a zero in the input is reproduced verbatim; the assert catches it at the edge that carries it rather than eight steps later.

That assert made the test fixture invalid: `A_10kX_weighted_network.pkl` held 80 zero-weight edges, which the Mapper duplicated into every graph it built. Replaced by `A_10kX_weighted_network_positive.pkl` — the same graph with those edges dropped and the largest connected component kept, 1816 of 1821 nodes, all weights positive.

Samples A–D encode "no width" as `1e-10` rather than `0`. That is a valid positive weight, the input data is kept as-is, and the code is required only to process it without failing — which it does. Verified end to end on all four samples: load, `Mapper`, and `analyze_graph` in both weighted and unweighted mode all complete with **zero NaNs**, and the generate round trip (`Mapper.assign_weights` sampling those values back into a fresh graph, then analysing it) also completes cleanly.

Worth recording, since it is invisible from the outside: those edges invert to a distance of 1e10 against a typical 21–36, so no shortest path routes through them. Dropping them would disconnect samples C and D (26 and 3 nodes) and take 36 more off B, so structurally they are load-bearing while metrically they are not — `is_connected()` reports `True` for C and D while distance-based metrics behave as if those nodes were detached. Sample B is already disconnected regardless.

### C.3 Eigenvector centrality was slow and imprecise — replaced

Profiling `analyze_graph()` on the 705-node fixture put **31.5s of 35.0s (90%) in `_compute_eigenvector_centrality`**. Everything else, curvature's 1052 `linprog` calls included, came to under 4s. The APSP redundancy in C.1 was never the bottleneck.

The cost came from `nk.centrality.EigenvectorCentrality(nk_graph, tol=1e-9)` — the "tighten the eigenvac" change. Tightening the tolerance was not buying accuracy, because networkit's implementation is power iteration and these graphs have a small spectral gap:

| tol | time | max relative diff vs 1e-12 |
| --- | --- | --- |
| 1e-12 | 34.0s | — |
| 1e-9 | 22.4s | 2.5% |
| 1e-8 | 19.1s | 8.3% |
| 1e-6 | 8.8s | 85% |

Replaced with `scipy.sparse.linalg.eigsh(A, k=1, which="LA")` (Lanczos) in `_principal_eigenvector`, unit-L2-normalised and made non-negative to match networkit's convention. scipy was already a dependency.

Timings: **0.024s vs 39.77s** unweighted (705 nodes), **0.012s vs 815.29s** weighted (1821 nodes).

**It is a speed fix, not a correctness fix — an earlier version of this note claimed otherwise and was wrong.** The two methods disagree by 1.13e-02 on the weighted fixture, and by residual `eigsh` is the accurate one (1.53e-15 against networkit's 5.73e-02 — networkit does not converge even at `tol=1e-12` after 815s):

```
eigsh (new code)        lambda=1.6231681994   residual=1.53e-15
networkit tol=1e-12     lambda=1.6221584027   residual=5.73e-02
```

But that error does not reach the reported numbers. Measured against the old `tol=1e-9` setting: `avg_eigen` moves **0.111%** unweighted and **0.198%** weighted, the top-100 nodes are identical in both, and Spearman rank correlation is 0.99999 unweighted. The weighted rank correlation over all nodes looks alarming at 0.356, but 1730 of that graph's 1821 nodes score below 1e-12 — numerically zero, so their ordering is noise. Restricted to the 91 nodes with meaningful scores, Spearman is **0.9999**.

So no previously saved result is invalidated. The justification for the change is 0.005s versus 27.5s for the same answers.

The regression test asserts the residual directly rather than agreement with networkit — pinning against a non-converged reference would pin the wrong answer.

Note the same weighted fixture has a smallest eigenvector score of `6.4e-21`, the near-detached cluster behind the C.2 crash. A tiny spectral gap is what makes power iteration crawl *and* the eigenvector ill-conditioned, so C.2 and C.3 are two symptoms of one structural quirk in that data.

### C.4 Where the time goes now (idle-machine audit)

After C.1 and C.3, `analyze_graph()` on the 705-node fixture is **2.473s, down from 35.0s — 14x**. The full test suite went from 205s to 98s as a side effect.

| component | time | share |
| --- | --- | --- |
| curvature (per-edge `linprog`) | 1.707s | 69% |
| multifractal taus | 0.279s | 11% |
| centralities | 0.124s | 5% |
| node dimension | 0.110s | 4% |
| betweenness | 0.042s | 2% |
| eigenvector (`eigsh`) | 0.005s | 0.2% |
| diameter | 0.004s | 0.2% |

**The remaining candidate is the transport solver.** `_wasserstein_lp` calls `scipy.optimize.linprog(method="highs")` once per edge, and that is now 69% of the analysis. POT's `ot.emd2` is a C network-simplex solver for exactly this problem and would plausibly bring `analyze_graph` near 1s. It needs a new dependency and has not been measured yet. GraphRicciCurvature is the reference implementation of the surrounding algorithm but is networkx-based, which is why this codebase has its own networkit-native version — the solver, though, did not need writing.

**Two candidates measured and rejected.** Vectorising `Mapper._compute_edge_metrics`'s per-edge `scipy.spatial.distance.euclidean` is a genuine 16x ratio but only **9ms → 1ms** on 2591 edges, once per network — not worth changing working code. `_compute_node_dimension`'s per-node `linregress` plus `sorted`/`Counter` is O(V² log V) of interpreter work, but costs 0.110s here; it only becomes interesting if graph sizes grow several-fold.

**What a user actually waits on is none of the above.** `ErrorChecker.check()` builds a fresh analyzer per candidate and calls `analyze_error_features()`, which runs one APSP and the taus — **no curvature, no eigenvector**. Measured at 0.159s (705 nodes, unweighted) and 0.877s (1821, weighted) per candidate. At 100 networks with up to 10 attempts each that is minutes of real waiting, and both components are already efficient. Generation time is governed by attempt counts and worker parallelism, not by micro-optimising these functions.

## Appendix F: forked workers spun forever on a many-core host — fixed

A `generate` run with the multifractal quality gate — the **default** — never finished. Not slowly: it never finished. Without the gate the same run took 2.5s.

networkit defaults to one thread per core, and this host has 128. The parent computes the error-checker reference, which starts that many OpenMP threads, then forks the worker pool. A forked child inherits an OpenMP runtime whose threads do not exist in it, and spins at 200% CPU instead of working. It is not a deadlock — an early note called it one and was wrong; the child burns CPU indefinitely.

Measured on `sample_A`, one network, one attempt:

| | result |
| --- | --- |
| fork, no pin | never finished (>40 min; a minimal repro was still spinning after 4 min) |
| fork, `nk.setNumberOfThreads(1)` in the child | **0.3s** |
| spawn | 4.3s |

Fixed by pinning the worker to one thread — exactly what `pipelines/hybrid.py` already did at lines 175 and 267. Hybrid had been fixed long ago; `generate`, `mosaic` and `sweep` never were. `generate_from_props` submits the same worker as `generate`, so it is covered too. `scaling` has no process pool. The plot pools render images and never touch networkit.

After the fix, all four settings complete in about three seconds: gate off 2.4s, gate on 3.0s, weighted 2.9s, weighted with plotting 2.9s.

Why it survived: severity scales with core count, so on a 4-core laptop it looks like ordinary sluggishness rather than a hang, and the test suite's pipeline tests use small graphs with the gate mostly off. `tests/test_worker_threads.py` now pins the behaviour for all four workers, and was checked to fail when the pin is removed.

Worth considering separately: `spawn` would make this class of bug impossible, and the config refactor already made the pipelines spawn-safe. That is a behaviour change on its own merits, not bundled here.

## Appendix D: `Saver` class state — resolved

The three stages below are **implemented**. Recorded because the root cause is worth remembering: `DISABLE_SAVING` was an ambient global, and both oddities in `Saver` were its shape, not independent design choices.

`cfg.disable_saving()` did not touch `cfg` — it set `BaseConfig.DISABLE_SAVING = True` for every config in the process, and for every forked child with it. Because no call site passed the flag, "saving is off" had to be visible everywhere, so it was encoded in the *existence of the object*: `Saver.__new__` returned `None` and `if not self.saver` became the disabled-check. That left no instance to call `begin_batch()` on, which forced the batch timestamp to be class state.

1. **`DISABLE_SAVING` is now a declared config attribute.** `enable_saving` / `disable_saving` are deleted — the last global class mutation in the codebase. Sweep declares `DISABLE_SAVING = True` on its own config; `scripts/helpers/reprocess_hybrid_lcc.py` and `scripts/runners/run_sweep.py` each declare a local subclass instead of mutating `BaseConfig`.
2. **`Saver.__new__` is deleted**, along with the `DISABLE_SAVING` assert and the early-return inside `save()`. A Saver that exists always writes. `RunAgent._build_saver` makes the decision in one visible place.
3. **The batch timestamp is instance state** and `begin_batch` / `end_batch` are instance methods, across 7 production and 3 test call sites. `tests/conftest.py::_reset_singletons` is gone — there is no class state left to reset.

Two things this surfaced. `run_sweep.py` read `BaseConfig.X` in ten places and only worked *because* of the mutation; it now reads its own config class. And `pipelines/hybrid.py` deletes `data_agent` before plotting to free memory, so its final `end_batch()` has to go through the local `saver` — flake8's `F821` caught that, not the test suite.

Still true, and deliberately not changed: `RunAgent.saver` is `None` when saving is disabled, so the call sites that reach through it (`data_agent.saver.save(...)` in mosaic, scaling, hybrid, hybrid_snapshot, generate_select) still raise `AttributeError` under `DISABLE_SAVING`. That was already the case before this work — no live path exercises them with saving off. Removing the `None` entirely means treating "do not save" as a destination rather than a mode: a dry run writes to a throwaway directory and the flag disappears. That trade costs real I/O during sweeps, which is what the flag exists to avoid, so it stays open.

## Appendix E: mypy measurement (notes only)

mypy 2.3.0, run once for measurement against `configs graphs handlers pipelines analysis gui run.py gui_run.py gui_app.py` with `--ignore-missing-imports`. Nothing was committed: no config file, no `.pre-commit-config.yaml` entry, no fixes.

**110 errors.** Almost none are bugs:

| code | n | what they actually are |
| --- | --- | --- |
| `attr-defined` | 39 | mostly `configs/enums.py`: `_levels` lives in `__slots__` and is set via `object.__setattr__`, which mypy cannot see. A single `_levels: Tuple[str, ...]` annotation silences ~9 of them. |
| `assignment` | 26 | implicit Optional — `def f(graph: SynthGraph = None)`. Mechanical PEP 484 style, spread across many files. |
| `arg-type` | 20 | `str \| None` config paths reaching `open`/`np.load`. **Inherent to the design**: paths are deliberately `None` until a config supplies them, so mypy wants a narrowing assert at every use. |
| other | 25 | `var-annotated`, `index`, `union-attr`, `no-redef`, small counts. |

Worst files: `utils.py` (24), `graphs/_graph_node.py` (18), `run.py` (15), `configs/enums.py` (9).

**Recommendation: do not adopt now.** The signal-to-noise is poor, and the largest real category argues *against* the type checker rather than for it — the fail-fast config design intentionally leaves attributes unset until a config provides them, and mypy reads exactly that as an error. Adoption would mean an implicit-Optional sweep plus narrowing asserts across the config surface, for no demonstrated bug caught.

`black` / `isort` / `flake8` already run in `.pre-commit-config.yaml` and pass; flake8 has caught real breakage in this codebase before. That gate is enough.

mypy was installed into `.venv` for this measurement and is not in `requirements.txt`. Remove with `.venv/bin/pip uninstall mypy` if unwanted.
