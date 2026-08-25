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

#: The frames that follow the input image until the user sets one.
_FRAME_KEYS = ("FRAME_SIZE", "SYNTHETIC_FRAME_SIZE")

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
        #: Whether the frames are still the image's, or the user's own.
        self._frame_touched = False

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

        # The form opens on the sample data, so say so and draw it rather than
        # waiting for an edit that has nothing to correct.
        self._note_ready()
        self._touch_preview()

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
            if section["name"]
            not in (spec_builder.OUTPUT_GROUP, spec_builder.ALIGN_GROUP)
        ]

    @Property("QVariantList", notify=changed)
    def alignLines(self) -> list:
        """The align section's rows — only for a single network.

        A folder holds networks that were traced from different images, and one
        turn applied to all of them would be right for at most one.  Editing is
        a thing you do to a network you are looking at.
        """
        if self._shapes()[self._shape].scope != spec_builder.SINGLE:
            return []
        return self._lines_of(spec_builder.ALIGN_GROUP)

    @Slot()
    def saveEdited(self) -> None:
        """Write the input as it is being read now, and read it back."""
        self._want("save_edited")

    @Property("QVariantList", notify=changed)
    def outputLines(self) -> list:
        """The output section's rows, for the pane that holds them."""
        return self._lines_of(spec_builder.OUTPUT_GROUP)

    def _lines_of(self, group: str) -> list:
        for section in self._sections():
            if section["name"] == group:
                return section["lines"]
        return []

    def _sections(self) -> list:
        """The mode's sections, each field carrying what it is set to now.

        The value travels in the model rather than being fetched by a slot: a
        binding on a slot call is evaluated once and never again, so a control
        went on showing whatever it was handed when it was built, however often
        the value behind it changed.
        """
        sections = spec_builder.sections_of(spec_builder.MODES[self._mode].fields)
        for section in sections:
            for line in section["lines"]:
                for spec_field in line["fields"]:
                    value = self._values.get(spec_field["id"], spec_field["value"])
                    spec_field["current"] = (
                        list(value) if isinstance(value, tuple) else value
                    )
        return sections

    @Property(bool, notify=changed)
    def hasInputChoice(self) -> bool:
        return len(spec_builder.MODES[self._mode].input_shapes) > 1

    @Property("QStringList", notify=changed)
    def inputScopes(self) -> list:
        """One network, or a folder of them — said apart from the format."""
        return [spec_builder.SCOPE_LABELS[scope] for scope in self._scopes()]

    @Property(int, notify=changed)
    def inputScope(self) -> int:
        return self._scopes().index(self._shapes()[self._shape].scope)

    @Slot(int)
    def selectInputScope(self, index: int) -> None:
        """Switch between one network and a folder, keeping the format if it
        exists on the other side.  A folder can only be read as CSV pairs, so
        coming back from one lands on whatever that scope does offer."""
        scopes = self._scopes()
        if not 0 <= index < len(scopes):
            return
        wanted = self._shapes()[self._shape].format
        for position, shape in enumerate(self._shapes()):
            if shape.scope == scopes[index] and shape.format == wanted:
                self.selectInputShape(position)
                return
        for position, shape in enumerate(self._shapes()):
            if shape.scope == scopes[index]:
                self.selectInputShape(position)
                return

    @Property("QStringList", notify=changed)
    def inputFormats(self) -> list:
        """The formats this scope can actually be read as."""
        scope = self._shapes()[self._shape].scope
        return [shape.format for shape in self._shapes() if shape.scope == scope]

    @Property(int, notify=changed)
    def inputFormat(self) -> int:
        return self.inputFormats.index(self._shapes()[self._shape].format)

    @Slot(int)
    def selectInputFormat(self, index: int) -> None:
        scope = self._shapes()[self._shape].scope
        matching = [
            position
            for position, shape in enumerate(self._shapes())
            if shape.scope == scope
        ]
        if 0 <= index < len(matching):
            self.selectInputShape(matching[index])

    def _shapes(self) -> list:
        return spec_builder.MODES[self._mode].input_shapes

    def _scopes(self) -> list:
        seen = []
        for shape in self._shapes():
            if shape.scope not in seen:
                seen.append(shape.scope)
        return seen

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
        if self._shapes()[index].scope != spec_builder.SINGLE:
            # The turn is not offered for a folder, so it must not linger in
            # the spec and quietly turn every network in one.
            self._values["INPUT_ORIENTATION"] = "none"
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
            rows.append(
                {
                    **spec_input.as_dict(),
                    # Shown short; stored, checked and run absolute.
                    "value": spec_builder.display_path(value),
                    "state": state,
                }
            )
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
        return spec_builder.display_path(self._output_dir)

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

    @Slot()
    def clearOutputs(self) -> None:
        """Write nothing beyond what a run cannot help writing.

        Every format off, and the plot count to zero with them: asking for
        images in no format is the one combination the form refuses, and
        "none" should not walk into it.
        """
        for spec_field in self._output_fields():
            self._values[spec_field.id] = False if spec_field.kind == "bool" else 0
        self._note_ready()

    @Slot()
    def resetOutputs(self) -> None:
        """Back to what the mode declares: networks as CSV, plots as WebP."""
        for spec_field in self._output_fields():
            self._values[spec_field.id] = spec_field.value
        self._note_ready()

    def _output_fields(self) -> list:
        """Whatever the output section holds — not a list repeated here."""
        return [
            spec_field
            for spec_field in spec_builder.MODES[self._mode].fields
            if spec_field.group == spec_builder.OUTPUT_GROUP
        ]

    @Slot(bool)
    def setShowBackground(self, on: bool) -> None:
        self._layers["background"] = on
        self.changed.emit()

    @Slot(bool)
    def setShowNetwork(self, on: bool) -> None:
        self._layers["network"] = on
        self.changed.emit()

    # ---- slots the form calls ----

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
        if key in _FRAME_KEYS:
            # From here the frame is theirs, and the image stops setting it.
            self._frame_touched = True
        self._touch_preview()

    @Slot(str, str)
    def setInput(self, key: str, value: str) -> None:
        self._inputs[key] = spec_builder.resolve_path(_local_path(value))
        self._touch_preview()
        self._note_ready()

    @Slot(str)
    def setOutputDir(self, value: str) -> None:
        self._output_dir = spec_builder.resolve_path(_local_path(value))
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
                # "choice" with them: it is a string from a fixed set, and
                # falling through to float() made every pick revert to the
                # default without a word.
                if spec_field.kind in ("text", "choice"):
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

    def _read_edited(self, info: dict) -> None:
        """Point the form at the copy that was just written.

        The turn goes back to none with it: the saved data already has it, and
        leaving it set would turn the network a second time on the next read.
        """
        shape = spec_builder.shape_for(self._mode, self._inputs)
        for key, path in info["inputs"].items():
            if key in shape.all_ids:
                self._inputs[key] = path
        self._values["INPUT_ORIENTATION"] = "none"
        self._info["original"] = None
        self._stale.add("original")
        self._set_status(f"Saved {info['count']} edited network(s) to {info['dir']}")

    def _adopt_image_size(self, info: dict) -> None:
        """Take the frames from the image the input came with.

        Only until the user sets one of their own: after that the image is
        just what gets drawn behind the network.  Re-renders once, because the
        picture that arrived was drawn in the frame this replaces.
        """
        size = info.get("image_size")
        if not size or self._frame_touched:
            return

        frame = (int(size[0]), int(size[1]))
        present = [key for key in _FRAME_KEYS if key in self._values]
        if all(self._values[key] == frame for key in present):
            return
        for key in present:
            self._values[key] = frame
        self._stale.add("original")

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
            kind = next(
                name
                for name in ("save_edited", "original", "synthetic")
                if name in self._stale
            )
            self._stale.discard(kind)
            # Saving is asked for outright; a preview only runs for a side that
            # is on screen to receive it.
            if kind == "save_edited" or self._shown[kind]:
                self._launch_preview(kind)
        self.changed.emit()

    def _launch_preview(self, kind: str) -> None:
        if kind in ("original", "save_edited"):
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
                info = json.load(handle)
            if self._preview_kind == "save_edited":
                self._read_edited(info)
            else:
                self._info[self._preview_kind] = info
                if self._preview_kind == "original":
                    self._adopt_image_size(info)
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
