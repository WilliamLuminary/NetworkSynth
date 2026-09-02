# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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

from networksynth.configs import BaseConfig
from networksynth.gui import spec_builder

_QML = os.path.join(os.path.dirname(__file__), "qml", "SynthesisWindow.qml")
_GUI_RUN = ["-m", "networksynth.gui_run"]
_GUI_PREVIEW = ["-m", "networksynth.gui_preview"]

_FRAME_KEYS = ("FRAME_SIZE", "SYNTHETIC_FRAME_SIZE")

_EXIT_MEANING = {
    0: "Finished.",
    1: "Run failed — see the log.",
    2: "The run-spec was rejected.",
    130: "Cancelled.",
    -9: "Cancelled — the run had to be forced.",
}

_PROCESS_GROUPS = hasattr(os, "killpg") and hasattr(os, "getpgid")

_CANCEL_GRACE = 15.0


def _local_path(value: str) -> str:
    if value.startswith("file://"):
        return QUrl(value).toLocalFile()
    return value


def _file_url(path: Optional[str]) -> str:
    return QUrl.fromLocalFile(path).toString() if path else ""


def _last_line(path: str) -> str:
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return lines[-1][:200] if lines else "no output"


class SynthesisController(QObject):

    changed = Signal()
    logChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None, *, mode: str = "generate"):
        super().__init__(parent)
        self._mode = mode
        self._values: Dict[str, Any] = spec_builder.default_values(mode)
        self._shape = 0
        self._inputs: Dict[str, str] = spec_builder.default_inputs(mode)
        self._output_dir = BaseConfig.BASE_OUTPUT_PATH
        self._status = "Choose this mode's inputs, then Run."
        self._failed = False
        self._percent = 0.0
        self._log: List[str] = []
        self._process: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self._log_offset = 0
        self._run_root: Optional[str] = None

        self._cancel_at: Optional[float] = None
        self._poll = QTimer(self)
        self._poll.setInterval(400)
        self._poll.timeout.connect(self._tick)

        self._spec_path: Optional[str] = None
        self._run_mode = ""
        self._ran_ok = False
        self._frame_touched = False

        self._shown = {"original": True, "synthetic": False}
        self._info: Dict[str, Optional[dict]] = {"original": None, "synthetic": None}
        self._layers = {"background": True, "network": True}

        self._stale: set = set()
        self._preview_process: Optional[subprocess.Popen] = None
        self._preview_kind = ""
        self._preview_out = ""
        self._preview_stderr: Optional[Any] = None
        self._scratch: Optional[str] = None
        self._preview_count = 0
        self._preview_error = ""

        self._preview_poll = QTimer(self)
        self._preview_poll.setInterval(300)
        self._preview_poll.timeout.connect(self._preview_tick)

        self._settle = QTimer(self)
        self._settle.setInterval(400)
        self._settle.setSingleShot(True)
        self._settle.timeout.connect(lambda: self._want("original"))

        self._note_ready()
        self._touch_preview()

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
        names = list(spec_builder.MODES)
        if not 0 <= index < len(names) or names[index] == self._mode:
            return
        self._mode = names[index]
        self._values = spec_builder.default_values(self._mode)
        self._shape = 0
        self._inputs = spec_builder.default_inputs(self._mode)
        self._info["original"] = None
        self._shown["synthetic"] = False
        self._info["synthetic"] = None
        self._note_ready()
        self._want("original")

    @Property("QVariantList", notify=changed)
    def configSections(self) -> list:
        return [
            section
            for section in self._sections()
            if section["name"]
            not in (spec_builder.OUTPUT_GROUP, spec_builder.ALIGN_GROUP)
        ]

    @Property("QVariantList", notify=changed)
    def alignLines(self) -> list:
        if self._shapes()[self._shape].scope != spec_builder.SINGLE:
            return []
        return self._lines_of(spec_builder.ALIGN_GROUP)

    @Slot()
    def saveEdited(self) -> None:
        self._want("save_edited")

    @Property("QVariantList", notify=changed)
    def outputLines(self) -> list:
        return [
            line
            for line in self._lines_of(spec_builder.OUTPUT_GROUP)
            if line["row"] not in spec_builder.FORMAT_ROWS
        ]

    @Property("QVariantList", notify=changed)
    def formatLines(self) -> list:
        return [
            line
            for line in self._lines_of(spec_builder.OUTPUT_GROUP)
            if line["row"] in spec_builder.FORMAT_ROWS
        ]

    def _lines_of(self, group: str) -> list:
        for section in self._sections():
            if section["name"] == group:
                return section["lines"]
        return []

    def _sections(self) -> list:
        fields = [
            spec_field
            for spec_field in spec_builder.MODES[self._mode].fields
            if self._applies(spec_field)
        ]
        sections = spec_builder.sections_of(fields)
        for section in sections:
            for line in section["lines"]:
                for spec_field in line["fields"]:
                    value = self._values.get(spec_field["id"], spec_field["value"])
                    spec_field["current"] = (
                        list(value) if isinstance(value, tuple) else value
                    )
        return sections

    def _applies(self, spec_field) -> bool:
        return all(
            self._values.get(other) not in hidden_by
            for other, hidden_by in spec_field.hide_when
        )

    @Property(bool, notify=changed)
    def hasInputChoice(self) -> bool:
        return len(spec_builder.MODES[self._mode].input_shapes) > 1

    @Property("QStringList", notify=changed)
    def inputScopes(self) -> list:
        return [spec_builder.SCOPE_LABELS[scope] for scope in self._scopes()]

    @Property(int, notify=changed)
    def inputScope(self) -> int:
        return self._scopes().index(self._shapes()[self._shape].scope)

    @Slot(int)
    def selectInputScope(self, index: int) -> None:
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
        shapes = spec_builder.MODES[self._mode].input_shapes
        if not 0 <= index < len(shapes) or index == self._shape:
            return
        self._shape = index
        self._inputs = spec_builder.default_inputs(self._mode, index)
        if self._shapes()[index].scope != spec_builder.SINGLE:
            self._values["INPUT_ORIENTATION"] = "none"
        self._info["original"] = None
        self._note_ready()
        self._want("original")

    @Property("QVariantList", notify=changed)
    def inputs(self) -> list:
        shape = spec_builder.MODES[self._mode].input_shapes[self._shape]
        rows = []
        for spec_input in shape.inputs:
            value = self._inputs.get(spec_input.id, "")
            state, _ = spec_builder.input_state(spec_input, value)
            rows.append(
                {
                    **spec_input.as_dict(),
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
        if self._process is not None:
            return False
        return not spec_builder.validate(
            self._mode, self._inputs, self._output_dir, self._values
        )

    @Property(str, notify=changed)
    def runLabel(self) -> str:
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

    @Property(bool, notify=changed)
    def canPreview(self) -> bool:
        shape = spec_builder.MODES[self._mode].input_shapes[self._shape]
        return bool(spec_builder.NETWORK_INPUTS & set(shape.ids))

    @Property(bool, notify=changed)
    def offersSyntheticPreview(self) -> bool:
        return self._mode == "generate"

    @Property(bool, notify=changed)
    def canPreviewSynthetic(self) -> bool:
        return (
            self.offersSyntheticPreview
            and self._ran_ok
            and self._run_mode == "generate"
            and self._process is None
            and self._run_root is not None
        )

    @Slot(int)
    def selectPreviewTab(self, index: int) -> None:
        kind = "synthetic" if index else "original"
        self._shown[kind] = True
        if self._info[kind] is None:
            self._want(kind)
        else:
            self.changed.emit()

    @Property(bool, notify=changed)
    def hasBackground(self) -> bool:
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
        for spec_field in self._output_fields():
            self._values[spec_field.id] = False if spec_field.kind == "bool" else 0
        self._note_ready()

    @Slot()
    def resetOutputs(self) -> None:
        for spec_field in self._output_fields():
            self._values[spec_field.id] = spec_field.value
        self._note_ready()

    def _output_fields(self) -> list:
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

    @Slot(str, "QVariant")
    def setValue(self, key: str, value) -> None:
        self._values[key] = self._coerce(key, value)
        self._touch_preview(redraw=self._group_of(key) != spec_builder.OUTPUT_GROUP)

    @Slot(str, int, "QVariant")
    def setSize(self, key: str, index: int, value) -> None:
        current = list(self._values.get(key) or (0, 0))
        as_float = self._kind_of(key) == "range"
        try:
            current[index] = float(value) if as_float else int(float(value))
        except (TypeError, ValueError):
            return
        self._values[key] = tuple(current)
        if key in _FRAME_KEYS:
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
            [sys.executable, *_GUI_RUN, spec_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=_PROCESS_GROUPS,
        )
        self._cancel_at = None
        self._poll.start()
        self._set_status("Running…")
        self.logChanged.emit()

    @Slot()
    def cancel(self) -> None:
        if self._process is None:
            return
        self._set_status("Cancelling…")
        # To the group, not to the one process: a hybrid run has a dataset
        # child, a worker pool and a manager under it, and signalling only the
        # top orphans the pool and hangs the run at exit.
        self._cancel_at = time.monotonic()
        self._stop_run(force=False)

    def _stop_run(self, force: bool) -> None:
        if self._process is None:
            return
        try:
            if _PROCESS_GROUPS:
                os.killpg(
                    os.getpgid(self._process.pid),
                    signal.SIGKILL if force else signal.SIGINT,
                )
            elif force:
                self._process.kill()
            else:
                self._process.send_signal(signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _kind_of(self, key: str) -> str:
        for spec_field in spec_builder.MODES[self._mode].fields:
            if spec_field.id == key:
                return spec_field.kind
        return ""

    def _group_of(self, key: str) -> str:
        for spec_field in spec_builder.MODES[self._mode].fields:
            if spec_field.id == key:
                return spec_field.group
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
                if spec_field.kind == "choice":
                    for option in spec_field.options:
                        if option is value or str(option) == str(value):
                            return option
                    return spec_field.value
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
            self._cancel_at = None
            self._poll.stop()
            self._drain_log()
            self._finish(code)
        elif (
            self._cancel_at is not None
            and time.monotonic() - self._cancel_at > _CANCEL_GRACE
        ):
            self._cancel_at = None
            self._log.append(
                f"{'WARNING':<8}Still running {_CANCEL_GRACE:.0f}s "
                "after cancelling. Forcing it to stop."
            )
            self.logChanged.emit()
            self._stop_run(force=True)
        self.changed.emit()

    def _drain_log(self) -> None:
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

    def _scratch_dir(self) -> str:
        if self._scratch is None:
            self._scratch = tempfile.mkdtemp(prefix="networksynth_gui_")
            atexit.register(shutil.rmtree, self._scratch, ignore_errors=True)
        return self._scratch

    def _write_spec(self, name: str) -> str:
        spec = spec_builder.build_spec(
            self._mode, self._inputs, self._output_dir, self._values
        )
        return spec_builder.write_spec(spec, os.path.join(self._scratch_dir(), name))

    def _text_of(self, kind: str) -> str:
        info = self._info[kind]
        return info["text"] if info else ""

    def _note_of(self, kind: str) -> str:
        info = self._info[kind]
        return info["note"] if info else ""

    def _read_edited(self, info: dict) -> None:
        shape = spec_builder.shape_for(self._mode, self._inputs)
        for key, path in info["inputs"].items():
            if key in shape.all_ids:
                self._inputs[key] = path
        self._values["INPUT_ORIENTATION"] = "none"
        self._info["original"] = None
        self._stale.add("original")
        self._set_status(f"Saved {info['count']} edited network(s) to {info['dir']}")

    def _adopt_image_size(self, info: dict) -> None:
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

    def _touch_preview(self, redraw: bool = True) -> None:
        self.changed.emit()
        if redraw and self._shown["original"]:
            self._settle.start()

    def _want(self, kind: str) -> None:
        self._stale.add(kind)
        self._pump_preview()

    def _pump_preview(self) -> None:
        while self._preview_process is None and self._stale:
            kind = next(
                name
                for name in ("save_edited", "original", "synthetic")
                if name in self._stale
            )
            self._stale.discard(kind)
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

        self._preview_count += 1
        out_dir = os.path.join(self._scratch_dir(), f"{kind}_{self._preview_count}")
        os.makedirs(out_dir)

        command = [sys.executable, *_GUI_PREVIEW, spec_path, kind, out_dir]
        if run_root:
            command.append(run_root)

        self._preview_kind = kind
        self._preview_out = out_dir
        self._preview_error = ""
        self._preview_stderr = open(os.path.join(out_dir, "stderr.txt"), "w")
        self._preview_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=self._preview_stderr,
            stdin=subprocess.DEVNULL,
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
