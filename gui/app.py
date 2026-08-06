# src/gui/app.py
"""The synthesis window: a form that launches a run and follows its progress.

Deliberately does **not** call the pipelines in-process.  It writes a run-spec,
starts ``gui_run.py`` as a subprocess, and tails the run's ``run.jsonl`` for
progress records.  That is exactly the mechanism StructuralGT's controller will
use, so this window doubles as its reference implementation — and it means a
crash or a cancel cannot take the UI down with it.
"""

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

#: Exit codes gui_run.py promises.
_EXIT_MEANING = {
    0: "Finished.",
    1: "Run failed — see the log.",
    2: "The run-spec was rejected.",
    130: "Cancelled.",
}


def _local_path(value: str) -> str:
    """QML file dialogs hand back file:// URLs; everything else is already a path."""
    if value.startswith("file://"):
        return QUrl(value).toLocalFile()
    return value


class SynthesisController(QObject):
    """Bridges the QML form to a ``gui_run.py`` subprocess."""

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
        self._edge_list = ""
        self._positions = ""
        self._output_dir = os.path.join(_REPO_ROOT, "data", "output")
        self._status = "Choose an edge list and positions file, then Run."
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

    # ---- properties bound by QML ----

    @Property(str, notify=changed)
    def mode(self) -> str:
        return self._mode

    @Property("QVariantList", notify=changed)
    def fields(self) -> list:
        return [f.as_dict() for f in spec_builder.MODES[self._mode].fields]

    @Property(str, notify=changed)
    def edgeList(self) -> str:
        return self._edge_list

    @Property(str, notify=changed)
    def positions(self) -> str:
        return self._positions

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

    # ---- slots the form calls ----

    @Slot(str, result="QVariant")
    def valueOf(self, key: str):
        value = self._values.get(key)
        return list(value) if isinstance(value, tuple) else value

    @Slot(str, "QVariant")
    def setValue(self, key: str, value) -> None:
        self._values[key] = self._coerce(key, value)

    @Slot(str, int, "QVariant")
    def setSize(self, key: str, index: int, value) -> None:
        current = list(self._values.get(key) or (0, 0))
        try:
            current[index] = int(float(value))
        except (TypeError, ValueError):
            return
        self._values[key] = tuple(current)

    @Slot(str)
    def setEdgeList(self, value: str) -> None:
        self._edge_list = _local_path(value)
        self._note_ready()

    @Slot(str)
    def setPositions(self, value: str) -> None:
        self._positions = _local_path(value)
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
            self._mode, self._edge_list, self._positions, self._output_dir, self._values
        )
        if problems:
            self._set_status("  •  ".join(problems), failed=True)
            return

        spec = spec_builder.build_spec(
            self._mode,
            self._edge_list,
            self._positions,
            self._output_dir,
            self._values,
        )
        spec_path = os.path.join(
            tempfile.mkdtemp(prefix="networksynth_gui_"), "run_spec.json"
        )
        spec_builder.write_spec(spec, spec_path)

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
            self._process.send_signal(
                getattr(__import__("signal"), "SIGINT", 2)
            )
        except (ProcessLookupError, OSError):
            pass

    # ---- internals ----

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

    def _read_manifest(self) -> Optional[dict]:
        if not self._run_root:
            return None
        path = os.path.join(self._run_root, "manifest.json")
        try:
            with open(path) as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _note_ready(self) -> None:
        problems = spec_builder.validate(
            self._mode, self._edge_list, self._positions, self._output_dir, self._values
        )
        self._set_status(
            "Ready." if not problems else problems[0], failed=False
        )

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
