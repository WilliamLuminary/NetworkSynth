# SPDX-License-Identifier: GPL-3.0-or-later
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
