# NetworkSynth → StructuralGT Integration Plan

Target: `structural-gt` v3.8.6 (`sgtlib`), branch `gen`.
Model: **loose coupling.** A button in their GUI launches NetworkSynth as a
separate process with its own environment. They never import our code.
Status: the groundwork on our side is built and tested (seeding, result
manifest, CSV reader, exit codes, structured progress, and a config refactor that
makes the pipelines work under a `spawn` start method). Nothing has been written
against StructuralGT itself, and the file contract in section 3 is not yet agreed
with their maintainer. See section 5 for what is done and what is outstanding.

---

## 1. Why loose coupling makes this easy

Deciding on a separate process rather than an in-process library removes most of
the hard problems before they start:

| Problem in an in-process merge | Under loose coupling |
|---|---|
| StructuralGT is on Python 3.14, networkit has no 3.14 wheels | **Gone.** We run in our own 3.12 environment. |
| `BaseConfig` mutates class-level global state | **Fine.** Every run is a fresh process — and this has since been removed anyway. |
| `BaseConfig._setup_logger()` hijacks the root logger | **Fine** for correctness, but the log lands in the wrong directory — see section 5. |
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
        spec = json.load(open(spec_path))
        cls.MODE = spec["mode"]
        for key, value in spec["params"].items():
            setattr(cls, key, value)          # FRAME_SIZE, factors, counts, SEED…
        cls.DATASETS = [DatasetId(spec.get("run_name", "gui_run"))]
        cls.BASE_OUTPUT_PATH = spec["output_dir"]
        cls._paths = spec["inputs"]

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
  "weight_type": "LEN",
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

**`weight_type` is not cosmetic.** Their `Weight` column comes from
`nx_graph[s][e]['weight']`, whose meaning depends on their `weight_type` setting
— diameter, area, length, angle, conductance or resistance. Ours comes from
`Mapper.assign_weights` and is a different quantity. Reading their weights works
mechanically, but the run-spec must record which quantity produced the file or a
synthetic network gets generated against something nobody tracked.

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

**Prerequisite:** the log currently lands in the repo's `data/output/logs/`
rather than inside `output_dir`, because `base_config.py` hardcodes
`BaseConfig.BASE_OUTPUT_PATH` for the logs directory. That must be fixed before
their controller can tail it — the run-spec names an `output_dir` and the log has
to be in it.

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

**Outstanding, and a prerequisite for the GUI.**

1. **Logging is misplaced.** `BaseConfig._setup_logger()` attaches handlers to the
   *root* logger and hardcodes `BaseConfig.BASE_OUTPUT_PATH` for the logs
   directory, so a run told to write to `output_dir` still logs into the repo.
   The plan has their controller tailing that file, so it has to land in the
   directory the run-spec named. A `_logger_initialized` latch also means a
   second run in one process gets no file handler at all. Logging is a process
   concern — it belongs in the entry points.

2. **Manifest for the remaining pipelines.** `generate` and `generate_select`
   write one; hybrid, hybrid_snapshot, mosaic, scaling and sweep do not yet.
   Only needed for whichever modes the GUI ends up exposing.

3. **Zero-weight edges.** 80 of 2591 edges in
   `tests/data/A_10kX_weighted_network.pkl` have weight exactly `0.0`.
   Legitimate, or an artifact of `Mapper.assign_weights`? Unresolved, and it
   interacts with `weight_type` in section 3.

4. **Redundant APSP** *(optional, performance).* In unweighted mode the analyzer
   computes the same all-pairs matrix twice (`multifractal_analyzer.py`);
   caching it roughly halves analysis cost. In weighted mode the two matrices
   genuinely differ, so the saving does not apply there.

---

## 6. Sequencing

| Phase | Work | Verify | State |
|---|---|---|---|
| 0 | Agree the file contract and the button's scope with their maintainer | written contract; no code | **their call** |
| 1 | `SEED` + thread it through | same seed twice → identical output | **done** |
| 2 | Result manifest | manifest lists every produced file; paths resolve | **done** (generate, generate_select) |
| 3 | CSV reader | reads their export format | **done** |
| 4 | Exit codes + structured progress | 130 on cancel; numeric `percent` in the log | **done** |
| 5 | Move logging to the entry points | log lands inside the run-spec's `output_dir` | todo — blocks 7 |
| 6 | `configs/gui_config.py` + run-spec loader + `gui_run.py` | a hand-written `run_spec.json` produces a network from CSV inputs, no GUI involved | todo |
| 7 | Their `synthesis_controller.py` + subprocess launch, headless | controller runs us end to end, parses the manifest, reads progress | todo |
| 8 | `SynthesisWindow.qml` + ribbon button + `[synthesis-settings]` ini section | click-through in the GUI; progress bar advances; Cancel terminates the child and yields `cancelled` | todo |
| 9 | Results re-enter via `add_graph()` | a generated network opens as a new `sgt_obj` and their GT PDF works on it | todo |

Phases 1–6 are entirely on our side. Phases 7–9 are theirs, and small.

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

Measured on `tests/data/A_10kX_weighted_network.pkl` (n=1821, e=2591) and
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

## Appendix C: measured issues in `analysis/multifractal_analyzer.py` (notes only)

Found while benchmarking; **nothing here is implemented**. Recorded so the
measurements are not lost. Each needs a yes before any code changes.

### C.1 A full analysis runs APSP three times

`analyze_graph()` computes all-pairs shortest paths three separate times over
the same distances:

| caller | graph |
| --- | --- |
| `_compute_multifractal_taus` (~line 284) | analysis graph |
| `_compute_node_dimension` (~line 370) | analysis graph (unweighted) / inverted (weighted) |
| `_ollivier_ricci_curvature` (~line 104) | builds its own copy of the same graph |

Measured on the unweighted 705-node fixture: **3 APSP runs, 48.5s**.

Collapsing them to one memoised call per distinct graph measured **26.0s — a
1.9x speedup** on `analyze_graph()`. Verified output-preserving: every
APSP-derived metric (`tau_list`, `dim_list`, `nfd_dist`, `ricci_dist`,
`closeness_dist`, `diameter`) was bit-identical. `betweenness_dist` and
`eigen_dist` differed at ~1e-16, but a control run of *unmodified* code against
itself differs by the same amount in the same two metrics — that is networkit's
threaded nondeterminism, not the cache.

Cost, and why it is not a several-line change: the first two call sites are a
~12-line memo, but the third is a module-level function that constructs its own
distance graph. Reusing the cache there means changing its signature to accept
`(dist_graph, dist_matrix)`. Two thirds of the win (3 runs → 2) is available
from the small change alone.

Memory: the memo holds a V×V float matrix per distinct graph, ~8 MB for
V=1000, ~800 MB for V=10000. Worth a size guard if scaling mode goes large.

### C.2 Weighted analysis crashes on zero-weight edges

Reproduced on the weighted fixture (1821 nodes, 2591 edges, **80 of them
zero-weight**):

```
ValueError: Invalid input for linprog: b_eq must not contain values inf, nan, or None
  analysis/multifractal_analyzer.py:150  _wasserstein_lp
```

Chain: `1/w` maps a zero weight to `float("inf")` (lines 95 and 226) → the
affinity `base ** (-inf**2)` underflows to 0 → when every neighbour of a node
arrives via such an edge, `w_u.sum()` is 0 → `0/0` produces NaN → the solver
rejects the problem.

The infinities also reach the distance matrix itself: **18,160 of 3,316,041
pairs are unreachable**, because a ~5-node cluster attaches to the rest of the
graph only through zero-weight edges. No node is isolated on its own — every
node touching a zero edge also has a finite edge. So the `1/w → inf` mapping
manufactures disconnection in a graph that is topologically connected, which
silently corrupts closeness, betweenness and node dimension in weighted mode
even where it does not crash.

Origin is upstream, not in the analyzer: `handlers/mapper.py:96` samples weights
from the *source* network's empirical distribution, so zeros in the input data
are reproduced faithfully. A zero weight is a zero-width wire.

Decision deferred by design. Per the current stance the run **should crash**
rather than paper over bad input; the open question is only whether the raw
scipy `ValueError` is a good enough report, or whether the crash should name the
offending edges. Any numeric treatment (flooring the weight, uniform fallback)
would silently alter the physics and is explicitly rejected for now.

Coverage gap worth noting on its own: no test exercises weighted analysis on
this fixture, which is why a hard crash on a supported code path went unnoticed.

## Appendix D: the last of `Saver`'s class state (notes only)

Stage 1 is done — `Saver` holds `(config, out_dir)` and the run directory
arrives as a `RunPaths` value. Two pieces remain, and they are **coupled**, so
neither is the small independent fix it looks like.

### D.1 `__new__` returns `None` when saving is disabled

```python
def __new__(cls, config, *args, **kwargs):
    if config.DISABLE_SAVING:
        return None          # constructor yields None
    return super().__new__(cls)
```

A constructor that returns `None` is a footgun, but it is at least *explicit*:
callers must write `if saver:`. The originally planned fix was a Null Object
whose `save()` does nothing — which removes the guard by making a disabled save
silently succeed. Under the current fail-loud stance that is the wrong
direction, so this needs a decision, not an implementation.

The fail-fast alternative worth considering instead: do not construct a `Saver`
at all when saving is disabled, and let the pipeline skip the save path
explicitly rather than routing through an object that discards its input.

### D.2 The batch timestamp is class state

`Saver._batch_timestamp` is a class attribute driven by the `begin_batch` /
`end_batch` classmethods — the last mutable class state in the codebase, and the
only reason `tests/conftest.py::_reset_singletons` still exists.

Instance-scoping it is mechanically small: all 11 call sites already have an
instance in scope (`data_agent.saver`, `self.saver`), so
`Saver.begin_batch()` becomes `data_agent.saver.begin_batch()`.

**Why it is blocked on D.1:** a classmethod works when no instance exists, which
is exactly the `DISABLE_SAVING` case. Turn it into an instance method and all 11
sites raise `AttributeError` on `None` under disabled saving, unless every one
of them grows a guard. So D.1 must be settled first.

Call sites, for whenever this is picked up: `pipelines/hybrid.py` (681, 692),
`pipelines/hybrid_snapshot.py` (227, 238), `pipelines/mosaic.py` (256),
`pipelines/scaling.py` (52), `handlers/run_agent.py` (132), plus
`tests/test_save_infrastructure.py` (245) and `tests/test_integration_modes.py`
(191, 244).

## Appendix E: mypy measurement (notes only)

mypy 2.3.0, run once for measurement against `configs graphs handlers pipelines
analysis gui run.py gui_run.py gui_app.py` with `--ignore-missing-imports`.
Nothing was committed: no config file, no `.pre-commit-config.yaml` entry, no
fixes.

**110 errors.** Almost none are bugs:

| code | n | what they actually are |
| --- | --- | --- |
| `attr-defined` | 39 | mostly `configs/enums.py`: `_levels` lives in `__slots__` and is set via `object.__setattr__`, which mypy cannot see. A single `_levels: Tuple[str, ...]` annotation silences ~9 of them. |
| `assignment` | 26 | implicit Optional — `def f(graph: SynthGraph = None)`. Mechanical PEP 484 style, spread across many files. |
| `arg-type` | 20 | `str \| None` config paths reaching `open`/`np.load`. **Inherent to the design**: paths are deliberately `None` until a config supplies them, so mypy wants a narrowing assert at every use. |
| other | 25 | `var-annotated`, `index`, `union-attr`, `no-redef`, small counts. |

Worst files: `utils.py` (24), `graphs/_graph_node.py` (18), `run.py` (15),
`configs/enums.py` (9).

**Recommendation: do not adopt now.** The signal-to-noise is poor, and the
largest real category argues *against* the type checker rather than for it —
the fail-fast config design intentionally leaves attributes unset until a config
provides them, and mypy reads exactly that as an error. Adoption would mean an
implicit-Optional sweep plus narrowing asserts across the config surface, for no
demonstrated bug caught.

`black` / `isort` / `flake8` already run in `.pre-commit-config.yaml` and pass;
flake8 has caught real breakage in this codebase before. That gate is enough.

mypy was installed into `.venv` for this measurement and is not in
`requirements.txt`. Remove with `.venv/bin/pip uninstall mypy` if unwanted.
