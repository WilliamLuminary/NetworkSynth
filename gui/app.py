from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from PySide6.QtCore import (
    Property,
    QObject,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from gui import spec_builder

_QML = os.path.join(os.path.dirname(__file__), "qml", "SynthesisWindow.qml")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUI_RUN = os.path.join(_REPO_ROOT, "gui_run.py")
_GUI_PREVIEW = os.path.join(_REPO_ROOT, "gui_preview.py")

#: Exit codes gui_run.py promises.
_EXIT_MEANING = {
    0: "Finished.",
    1: "Run failed — see the log.",
    2: "The run-spec was rejected.",
    130: "Cancelled.",
}


def _local_path(value: str) -> str:
    if value.startswith("file://"):
        return QUrl(value).toLocalFile()
    return value


def _file_url(path: Optional[str]) -> str:
    return QUrl.fromLocalFile(path).toString() if path else ""


def _last_line(path: str) -> str:
    """The tail of a failed render's stderr, short enough to show in the form."""
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return lines[-1][:200] if lines else "no output"


class SynthesisController(QObject):

    changed = Signal()
    logChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None, *, mode: str = "generate"):
        # `parent` MUST be the first positional parameter.  With any extra
        # positional argument ahead of it, PySide6 still builds a valid QObject
        # — metaObject and Python access work — but QML resolves the context
        # property to null and every binding silently fails.
        super().__init__(parent)
        self._mode = mode
        self._values: Dict[str, Any] = spec_builder.default_values(mode)
        self._shape = 0
        self._inputs: Dict[str, str] = spec_builder.default_inputs(mode)
        self._output_dir = os.path.join(_REPO_ROOT, "data", "output")
        self._status = "Choose this mode's inputs, then Run."
        self._failed = False
        self._percent = 0.0
        self._log: List[str] = []
        self._process: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self._log_offset = 0
        self._run_root: Optional[str] = None

        self._poll = QTimer(self)
        self._poll.setInterval(400)
        self._poll.timeout.connect(self._tick)

        # What the last run was, so a preview of its output is offered only
        # when there is output of that kind to read.
        self._spec_path: Optional[str] = None
        self._run_mode = ""
        self._ran_ok = False

        # Previews.  Each side is shown or not, and holds the info.json its
        # render wrote; the layer switches choose between the original's three
        # variants, which are all rendered at once.
        self._shown = {"original": True, "synthetic": False}
        self._info: Dict[str, Optional[dict]] = {"original": None, "synthetic": None}
        self._layers = {"background": True, "network": True}

        # One render at a time, and a side asked for while one is running is
        # remembered rather than dropped.
        self._stale: set = set()
        self._preview_process: Optional[subprocess.Popen] = None
        self._preview_kind = ""
        self._preview_out = ""
        self._preview_stderr: Optional[Any] = None
        self._preview_root: Optional[str] = None
        self._preview_count = 0
        self._preview_error = ""

        self._preview_poll = QTimer(self)
        self._preview_poll.setInterval(300)
        self._preview_poll.timeout.connect(self._preview_tick)

        # Editing a field fires one change per field; a single render after the
        # typing stops beats one render per keystroke.
        self._settle = QTimer(self)
        self._settle.setInterval(400)
        self._settle.setSingleShot(True)
        self._settle.timeout.connect(lambda: self._want("original"))

    # ---- properties bound by QML ----

    @Property(str, notify=changed)
    def mode(self) -> str:
        return self._mode

    @Property("QStringList", notify=changed)
    def modes(self) -> list:
        return list(spec_builder.MODES)

    @Property("QStringList", notify=changed)
    def modeLabels(self) -> list:
        return [spec_builder.MODES[name].label for name in spec_builder.MODES]

    @Slot(int)
    def selectMode(self, index: int) -> None:
        """Switch mode and reload its fields.

        Values are rebuilt from the new mode's defaults rather than carried over:
        modes share names for the common fields but not for their own, and a
        stale value from another mode would be written into the run-spec and
        silently set on the config.
        """
        names = list(spec_builder.MODES)
        if not 0 <= index < len(names) or names[index] == self._mode:
            return
        self._mode = names[index]
        self._values = spec_builder.default_values(self._mode)
        self._shape = 0
        self._inputs = spec_builder.default_inputs(self._mode)
        self._info["original"] = None
        # The other mode's run is not this mode's output.
        self._shown["synthetic"] = False
        self._info["synthetic"] = None
        self._note_ready()

    @Property("QVariantList", notify=changed)
    def configSections(self) -> list:
        """The parameter sections the form pane shows, in order.

        Everything except the output section, which the window gives a pane of
        its own beside the preview: one side is what a run is given, the other
        is what it produces.
        """
        return [
            section
            for section in self._sections()
            if section["name"] != spec_builder.OUTPUT_GROUP
        ]

    @Property("QVariantList", notify=changed)
    def outputLines(self) -> list:
        """The output section's rows, for the pane that holds them."""
        for section in self._sections():
            if section["name"] == spec_builder.OUTPUT_GROUP:
                return section["lines"]
        return []

    def _sections(self) -> list:
        return spec_builder.sections_of(spec_builder.MODES[self._mode].fields)

    @Property("QStringList", notify=changed)
    def inputShapes(self) -> list:
        return [shape.label for shape in spec_builder.MODES[self._mode].input_shapes]

    @Property(int, notify=changed)
    def inputShape(self) -> int:
        return self._shape

    @Slot(int)
    def selectInputShape(self, index: int) -> None:
        """Switch between a single network and a directory of them.

        The keys change with the shape, so the values are rebuilt rather than
        carried over: a path left behind from the other shape would be written
        into the run-spec and reach a loader that never asked for it.
        """
        shapes = spec_builder.MODES[self._mode].input_shapes
        if not 0 <= index < len(shapes) or index == self._shape:
            return
        self._shape = index
        self._inputs = spec_builder.default_inputs(self._mode, index)
        self._info["original"] = None
        self._note_ready()

    @Property("QVariantList", notify=changed)
    def inputs(self) -> list:
        """What this mode reads, with the paths chosen so far and how they stand.

        A list rather than fixed properties: analysis takes a directory of
        finished results where generation takes a network's two CSVs, and a form
        offering the wrong one is a run that fails after it starts.
        """
        shape = spec_builder.MODES[self._mode].input_shapes[self._shape]
        rows = []
        for spec_input in shape.inputs:
            value = self._inputs.get(spec_input.id, "")
            state, _ = spec_builder.input_state(spec_input, value)
            rows.append({**spec_input.as_dict(), "value": value, "state": state})
        return rows

    @Property(str, notify=changed)
    def modeBlurb(self) -> str:
        return spec_builder.MODES[self._mode].blurb

    @Property(bool, notify=changed)
    def canRun(self) -> bool:
        """Whether the form is complete enough to start.

        The button goes dead rather than accepting a click it would only
        refuse; what is missing is named in the status bar either way.
        """
        if self._process is not None:
            return False
        return not spec_builder.validate(
            self._mode, self._inputs, self._output_dir, self._values
        )

    @Property(str, notify=changed)
    def runLabel(self) -> str:
        """Which run the window is showing, for the status bar.

        Read off the directory name the run made, which already carries both:
        ``gui_generate_results_<date>_<time>_<id>``.
        """
        if not self._run_root:
            return ""
        parts = os.path.basename(self._run_root).split("_")
        if len(parts) < 3:
            return ""
        stamp = parts[-2]
        clock = f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}" if len(stamp) == 6 else stamp
        return f"run {parts[-1]}  ·  {clock}"

    @Property(str, notify=changed)
    def outputDir(self) -> str:
        return self._output_dir

    @Property(str, notify=changed)
    def status(self) -> str:
        return self._status

    @Property(bool, notify=changed)
    def failed(self) -> bool:
        return self._failed

    @Property(bool, notify=changed)
    def running(self) -> bool:
        return self._process is not None

    @Property(float, notify=changed)
    def percent(self) -> float:
        return self._percent

    @Property(str, notify=logChanged)
    def logText(self) -> str:
        return "\n".join(self._log[-400:])

    # ---- preview ----

    @Property(bool, notify=changed)
    def canPreview(self) -> bool:
        """Whether this mode reads a network of its own to show."""
        shape = spec_builder.MODES[self._mode].input_shapes[self._shape]
        return bool({"edge_list", "datasets_dir"} & set(shape.all_ids))

    @Property(bool, notify=changed)
    def offersSyntheticPreview(self) -> bool:
        """Whether this mode's output is worth previewing at all.

        ``generate`` only.  A hybrid network is assembled from thousands of
        tiles, so drawing one is a long job with an unreadable result — and
        hybrid and sweep write no batch of networks for a preview to read
        either.
        """
        return self._mode == "generate"

    @Property(bool, notify=changed)
    def canPreviewSynthetic(self) -> bool:
        """Whether a finished run of that mode actually left something to show."""
        return (
            self.offersSyntheticPreview
            and self._ran_ok
            and self._run_mode == "generate"
            and self._process is None
            and self._run_root is not None
        )

    @Slot(int)
    def selectPreviewTab(self, index: int) -> None:
        """Show one side of the preview, and render it if it is not drawn yet."""
        kind = "synthetic" if index else "original"
        self._shown[kind] = True
        if self._info[kind] is None:
            self._want(kind)
        else:
            self.changed.emit()

    @Property(bool, notify=changed)
    def hasBackground(self) -> bool:
        """Whether the previewed input came with an image behind it.

        Answered by the render, not guessed here: which file counts as a
        dataset's image is the config's rule, and asking twice invites the two
        answers to differ.
        """
        info = self._info["original"]
        return bool(info and info["has_background"])

    @Property(bool, notify=changed)
    def showBackground(self) -> bool:
        return self._layers["background"]

    @Property(bool, notify=changed)
    def showNetwork(self) -> bool:
        return self._layers["network"]

    @Property(str, notify=changed)
    def originalImage(self) -> str:
        """The variant the two layer switches select, as a URL for QML.

        Empty when both are off, or when there is nothing rendered yet.
        """
        info = self._info["original"]
        if not info:
            return ""
        images = info["images"]
        if self._layers["network"]:
            key = (
                "both" if self._layers["background"] and "both" in images else "network"
            )
        elif self._layers["background"]:
            key = "background"
        else:
            return ""
        return _file_url(images.get(key))

    @Property(str, notify=changed)
    def originalInfo(self) -> str:
        return self._text_of("original")

    @Property(str, notify=changed)
    def originalNote(self) -> str:
        return self._note_of("original")

    @Property(str, notify=changed)
    def syntheticImage(self) -> str:
        info = self._info["synthetic"]
        return _file_url(info["images"]["network"]) if info else ""

    @Property(str, notify=changed)
    def syntheticInfo(self) -> str:
        return self._text_of("synthetic")

    @Property(str, notify=changed)
    def syntheticNote(self) -> str:
        return self._note_of("synthetic")

    @Property(str, notify=changed)
    def previewStatus(self) -> str:
        if self._preview_error:
            return self._preview_error
        if self._preview_process is not None:
            return "Rendering preview…"
        return ""

    @Property(bool, notify=changed)
    def previewFailed(self) -> bool:
        return bool(self._preview_error)

    @Slot(bool)
    def setShowBackground(self, on: bool) -> None:
        self._layers["background"] = on
        self.changed.emit()

    @Slot(bool)
    def setShowNetwork(self, on: bool) -> None:
        self._layers["network"] = on
        self.changed.emit()

    # ---- slots the form calls ----

    @Slot(str, result="QVariant")
    def valueOf(self, key: str):
        value = self._values.get(key)
        return list(value) if isinstance(value, tuple) else value

    @Slot(str, "QVariant")
    def setValue(self, key: str, value) -> None:
        self._values[key] = self._coerce(key, value)
        self._touch_preview()

    @Slot(str, int, "QVariant")
    def setSize(self, key: str, index: int, value) -> None:
        """One half of a two-part field: a frame size, or a swept range."""
        current = list(self._values.get(key) or (0, 0))
        as_float = self._kind_of(key) == "range"
        try:
            current[index] = float(value) if as_float else int(float(value))
        except (TypeError, ValueError):
            return
        self._values[key] = tuple(current)
        self._touch_preview()

    @Slot(str, str)
    def setInput(self, key: str, value: str) -> None:
        self._inputs[key] = _local_path(value)
        self._touch_preview()
        self._note_ready()

    @Slot(str)
    def setOutputDir(self, value: str) -> None:
        self._output_dir = _local_path(value)
        self.changed.emit()

    @Slot()
    def run(self) -> None:
        if self._process is not None:
            return

        problems = spec_builder.validate(
            self._mode, self._inputs, self._output_dir, self._values
        )
        if problems:
            self._set_status("  •  ".join(problems), failed=True)
            return

        spec_path = self._write_spec("run_spec.json")
        # Remembered so a preview of this run's output reads the same spec the
        # run did, whatever the form says by the time it finishes.
        self._spec_path = spec_path
        self._run_mode = self._mode
        self._ran_ok = False

        self._log = [f"run-spec: {spec_path}"]
        self._percent = 0.0
        self._failed = False
        self._log_path = None
        self._log_offset = 0
        self._run_root = None
        self._before = self._existing_run_dirs()

        self._process = subprocess.Popen(
            [sys.executable, _GUI_RUN, spec_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Closed, not inherited: a pipeline that asks for input (wandb
            # prompting for an API key) would otherwise stall the run forever
            # against a terminal the user is not looking at.
            stdin=subprocess.DEVNULL,
            cwd=_REPO_ROOT,
        )
        self._poll.start()
        self._set_status("Running…")
        self.logChanged.emit()

    @Slot()
    def cancel(self) -> None:
        """SIGINT, so the run shuts down the way Ctrl-C would and exits 130."""
        if self._process is None:
            return
        self._set_status("Cancelling…")
        try:
            self._process.send_signal(getattr(__import__("signal"), "SIGINT", 2))
        except (ProcessLookupError, OSError):
            pass

    # ---- internals ----

    def _kind_of(self, key: str) -> str:
        for spec_field in spec_builder.MODES[self._mode].fields:
            if spec_field.id == key:
                return spec_field.kind
        return ""

    def _coerce(self, key: str, value):
        for spec_field in spec_builder.MODES[self._mode].fields:
            if spec_field.id != key:
                continue
            try:
                if spec_field.kind == "integer":
                    return int(float(value))
                if spec_field.kind == "bool":
                    return bool(value)
                if spec_field.kind == "text":
                    return str(value)
                return float(value)
            except (TypeError, ValueError):
                return spec_field.value
        return value

    def _existing_run_dirs(self) -> set:
        try:
            return set(os.listdir(self._output_dir))
        except OSError:
            return set()

    def _find_run_root(self) -> Optional[str]:
        try:
            fresh = set(os.listdir(self._output_dir)) - self._before
        except OSError:
            return None
        for name in sorted(fresh):
            if "_results_" in name:
                return os.path.join(self._output_dir, name)
        return None

    def _tick(self) -> None:
        if self._run_root is None:
            self._run_root = self._find_run_root()
            if self._run_root:
                self._log_path = os.path.join(self._run_root, "run.jsonl")

        self._drain_log()

        if self._process is not None and self._process.poll() is not None:
            code = self._process.returncode
            self._process = None
            self._poll.stop()
            self._drain_log()
            self._finish(code)
        self.changed.emit()

    def _drain_log(self) -> None:
        """Read new JSON lines; each is one record, so partial writes are safe."""
        if not self._log_path or not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path) as handle:
                handle.seek(self._log_offset)
                new = handle.read()
                self._log_offset = handle.tell()
        except OSError:
            return

        appended = False
        for line in new.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "percent" in entry:
                self._percent = float(entry["percent"])
            self._log.append(f"{entry.get('level', ''):<8}{entry.get('message', '')}")
            appended = True
        if appended:
            self.logChanged.emit()

    def _finish(self, code: int) -> None:
        message = _EXIT_MEANING.get(code, f"Exited with code {code}.")
        manifest = self._read_manifest()
        if manifest:
            message += f"  {manifest['file_count']} files in {self._run_root}"
        self._set_status(message, failed=code not in (0, 130))
        if code == 0:
            self._percent = 100.0
            self._ran_ok = True
            if self._shown["synthetic"]:
                self._want("synthetic")

    def _read_manifest(self) -> Optional[dict]:
        if not self._run_root:
            return None
        path = os.path.join(self._run_root, "manifest.json")
        try:
            with open(path) as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_spec(self, name: str) -> str:
        """This form as a run-spec, on disk where a subprocess can read it."""
        spec = spec_builder.build_spec(
            self._mode, self._inputs, self._output_dir, self._values
        )
        return spec_builder.write_spec(
            spec, os.path.join(tempfile.mkdtemp(prefix="networksynth_gui_"), name)
        )

    def _text_of(self, kind: str) -> str:
        info = self._info[kind]
        return info["text"] if info else ""

    def _note_of(self, kind: str) -> str:
        info = self._info[kind]
        return info["note"] if info else ""

    def _touch_preview(self) -> None:
        """The form changed, so a shown preview is of something else now.

        Re-rendered once the edits settle rather than corrected on the spot:
        every field reports separately, and a render per keystroke would be
        most of them wasted.
        """
        if self._shown["original"]:
            self._settle.start()

    def _want(self, kind: str) -> None:
        self._stale.add(kind)
        self._pump_preview()

    def _pump_preview(self) -> None:
        """Start the next render, one at a time.

        A side asked for while another is rendering waits here instead of
        cancelling it, so turning both on shows both.
        """
        while self._preview_process is None and self._stale:
            kind = "original" if "original" in self._stale else "synthetic"
            self._stale.discard(kind)
            if self._shown[kind]:
                self._launch_preview(kind)
        self.changed.emit()

    def _launch_preview(self, kind: str) -> None:
        if kind == "original":
            problems = spec_builder.validate(
                self._mode, self._inputs, self._output_dir, self._values
            )
            if problems:
                self._preview_error = problems[0]
                return
            spec_path, run_root = self._write_spec("preview_spec.json"), None
        else:
            if not self.canPreviewSynthetic:
                self._preview_error = "Nothing generated yet to show."
                return
            spec_path, run_root = self._spec_path, self._run_root

        if self._preview_root is None:
            self._preview_root = tempfile.mkdtemp(prefix="networksynth_preview_")
        # A fresh directory per render, so a new picture never arrives at a path
        # QML has already cached an old one for.
        self._preview_count += 1
        out_dir = os.path.join(self._preview_root, f"{kind}_{self._preview_count}")
        os.makedirs(out_dir)

        command = [sys.executable, _GUI_PREVIEW, spec_path, kind, out_dir]
        if run_root:
            command.append(run_root)

        self._preview_kind = kind
        self._preview_out = out_dir
        self._preview_error = ""
        # Kept rather than discarded: a failed render's traceback is the only
        # account of why, and the form shows its last line.
        self._preview_stderr = open(os.path.join(out_dir, "stderr.txt"), "w")
        self._preview_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=self._preview_stderr,
            stdin=subprocess.DEVNULL,
            cwd=_REPO_ROOT,
        )
        self._preview_poll.start()

    def _preview_tick(self) -> None:
        process = self._preview_process
        if process is None or process.poll() is None:
            return

        code = process.returncode
        self._preview_process = None
        self._preview_poll.stop()
        self._preview_stderr.close()

        if code == 0:
            # Exit 0 promises info.json, so a missing one is a bug worth
            # hearing about rather than an empty panel.
            with open(os.path.join(self._preview_out, "info.json")) as handle:
                self._info[self._preview_kind] = json.load(handle)
            self._preview_error = ""
        else:
            self._info[self._preview_kind] = None
            self._preview_error = (
                f"Preview failed: {_last_line(self._preview_stderr.name)}"
            )

        self._pump_preview()

    def _note_ready(self) -> None:
        problems = spec_builder.validate(
            self._mode, self._inputs, self._output_dir, self._values
        )
        self._set_status("Ready." if not problems else problems[0], failed=False)

    def _set_status(self, text: str, failed: bool = False) -> None:
        self._status = text
        self._failed = failed
        self.changed.emit()


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    app = QGuiApplication(argv)
    engine = QQmlApplicationEngine()
    controller = SynthesisController()
    engine.rootContext().setContextProperty("controller", controller)
    engine.load(QUrl.fromLocalFile(_QML))
    if not engine.rootObjects():
        print("Failed to load the QML window", file=sys.stderr)
        return 1
    return app.exec()
