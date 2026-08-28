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

    patched = False
    for target in ("_init_config", "create_run_paths", "run"):
        if hasattr(module, target):
            monkeypatch.setattr(module, target, boom)
            patched = True
            break
    assert patched, f"no interception point found in pipelines.{module_name}"

    with pytest.raises(KeyboardInterrupt):
        module.main()


_A_CONFIG = "configs/generate_mode/config_sample.py"


def _pipeline(monkeypatch, behaviour):
    import run

    module = type(sys)("fake_pipeline")
    module.main = behaviour
    monkeypatch.setattr(run, "load_pipeline", lambda mode: module)
    return run


def test_entry_point_maps_interruption_to_130(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    run = _pipeline(monkeypatch, interrupted)

    assert run.main(["run.py", _A_CONFIG]) == 130


def test_entry_point_maps_failure_to_1(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("boom")

    run = _pipeline(monkeypatch, failed)

    assert run.main(["run.py", _A_CONFIG]) == 1


def test_entry_point_leaves_success_alone(monkeypatch):
    run = _pipeline(monkeypatch, lambda *a, **k: None)

    assert run.main(["run.py", _A_CONFIG]) == 0


def test_a_bad_argument_count_is_rejected_with_2():
    import run

    assert run.main(["run.py"]) == 2
    assert run.main(["run.py", "a", "b"]) == 2


def test_an_unknown_config_path_is_rejected_with_2():
    import run

    assert run.main(["run.py", "configs/generate_mode/config_nope.py"]) == 2


def test_a_config_without_a_mode_is_rejected_with_2(monkeypatch, tmp_path):
    import run
    from configs import BaseConfig

    class Modeless(BaseConfig):
        pass

    monkeypatch.setattr(run, "load_config", lambda path: Modeless)

    assert run.main(["run.py", _A_CONFIG]) == 2


@pytest.mark.slow
def test_real_sigint_exits_130():
    import os
    import signal
    import time

    proc = subprocess.Popen(
        [sys.executable, "run.py", "configs/generate_mode/config_snapshot.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        start_new_session=True,
    )
    try:
        time.sleep(8)
        if proc.poll() is not None:
            pytest.skip(f"run exited on its own with {proc.returncode}")
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        returncode = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

    assert returncode != 0, "cancelled run reported success"
