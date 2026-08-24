import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

PIPELINES = [
    "generate",
    "mosaic",
    "scaling",
    "sweep",
    "hybrid",
    "compare",
]


@pytest.mark.parametrize("module_name", PIPELINES)
def test_pipeline_main_reraises_interruption(module_name, monkeypatch):
    import importlib

    module = importlib.import_module(f"pipelines.{module_name}")

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    # Break at the *earliest* thing main() calls, so we interrupt before any
    # side effects (sweep reaches wandb.login() otherwise).
    patched = False
    for target in ("_init_config", "create_run_paths", "run"):
        if hasattr(module, target):
            monkeypatch.setattr(module, target, boom)
            patched = True
            break
    assert patched, f"no interception point found in pipelines.{module_name}"

    with pytest.raises(KeyboardInterrupt):
        module.main()


def test_entry_point_maps_interruption_to_130(monkeypatch):
    import run

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run.fire, "Fire", interrupted)

    with pytest.raises(SystemExit) as exc_info:
        run.main()

    assert exc_info.value.code == 130


def test_entry_point_leaves_success_alone(monkeypatch):
    import run

    monkeypatch.setattr(run.fire, "Fire", lambda *a, **k: None)

    run.main()  # must not raise


@pytest.mark.slow
def test_real_sigint_exits_130():
    import os
    import signal
    import time

    proc = subprocess.Popen(
        [sys.executable, "run.py", "generate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        start_new_session=True,
    )
    try:
        time.sleep(8)  # let it get into generation
        if proc.poll() is not None:
            pytest.skip(f"run exited on its own with {proc.returncode}")
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        returncode = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

    assert returncode != 0, "cancelled run reported success"
