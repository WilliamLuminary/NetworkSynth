# tests/test_spawn_safety.py
"""
Guards the property the config refactor exists to provide: worker processes get
their parameters explicitly, so the pipelines work under a ``spawn`` start
method — the default on macOS and Windows.

Before the refactor a child inherited the parent's mutated ``BaseConfig`` under
``fork``.  Under ``spawn`` it re-imported ``configs`` fresh and saw
``CLOSED_NODES_FACTOR`` missing (``AttributeError``) and ``FULL_Q_BAND`` as a
plausible-but-wrong ``False`` — silently analysing with the wrong q-band rather
than failing.

The pipeline runs in a separate interpreter because ``set_start_method`` is
process-global and would otherwise change the whole test session.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_fixture_data,
    pytest.mark.slow,
]

_RUNNER = os.path.join(os.path.dirname(__file__), "_spawn_pipeline.py")


def test_generate_pipeline_runs_under_spawn():
    """A pipeline whose workers rely on inherited class state fails here."""
    result = subprocess.run(
        [sys.executable, _RUNNER],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        "generate pipeline failed under spawn:\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )
    assert "files_written=" in result.stdout
    written = int(result.stdout.split("files_written=")[1].split()[0])
    assert written > 0, "ran under spawn but produced no output"


def test_spawned_child_does_not_inherit_config_state():
    """Sanity check on the test above: the child really is a fresh interpreter.

    If a spawned child saw the parent's mutated BaseConfig, the pipeline passing
    would prove nothing.
    """
    program = (
        "import multiprocessing as mp, sys, json\n"
        f"sys.path.insert(0, {os.path.join(os.path.dirname(__file__), '..')!r})\n"
        "def child(q):\n"
        "    from configs import BaseConfig\n"
        "    q.put(getattr(BaseConfig, 'CLOSED_NODES_FACTOR', '<unset>'))\n"
        "if __name__ == '__main__':\n"
        "    mp.set_start_method('spawn', force=True)\n"
        "    from configs.generate_mode.config_sample import SampleConfig\n"
        "    SampleConfig.initialize()\n"
        "    q = mp.Queue(); p = mp.Process(target=child, args=(q,))\n"
        "    p.start(); p.join(60)\n"
        "    print('child_saw=' + str(q.get(timeout=5)))\n"
        "    print('parent_had=' + str(SampleConfig.CLOSED_NODES_FACTOR))\n"
    )
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=300
        )
    finally:
        os.unlink(path)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "child_saw=<unset>" in result.stdout, (
        "child inherited config state, so the spawn test above is not meaningful:\n"
        f"{result.stdout}"
    )
    assert "parent_had=" in result.stdout
