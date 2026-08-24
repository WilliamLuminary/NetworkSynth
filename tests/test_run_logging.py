import json
import logging
import os

import pytest

from handlers.run_logging import (
    attach_run_log,
    configure_console,
    reset_logging,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_logging():
    reset_logging()
    yield
    reset_logging()


def _our_handlers():
    return [
        h
        for h in logging.getLogger().handlers
        if getattr(h, "_networksynth_console", False)
        or getattr(h, "_networksynth_run_log", False)
    ]


class TestConsole:
    def test_attaches_one_handler(self):
        configure_console()

        assert len(_our_handlers()) == 1

    def test_is_idempotent(self):
        configure_console()
        configure_console()
        configure_console()

        assert len(_our_handlers()) == 1

    def test_repeat_call_adjusts_level(self):
        configure_console(logging.INFO)
        configure_console(logging.DEBUG)

        handler = _our_handlers()[0]
        assert handler.level == logging.DEBUG


class TestRunLog:
    def test_writes_inside_the_run_directory(self, tmp_path):
        path = attach_run_log(str(tmp_path), "abc123")

        assert path == os.path.join(str(tmp_path), "run.jsonl")
        assert os.path.dirname(path) == str(tmp_path)

    def test_records_carry_the_run_id(self, tmp_path):
        attach_run_log(str(tmp_path), "deadbeef")
        logging.getLogger("test").info("hello")

        entries = [
            json.loads(line)
            for line in open(os.path.join(str(tmp_path), "run.jsonl"))
            if line.strip()
        ]
        assert entries
        assert entries[-1]["run_id"] == "deadbeef"

    def test_second_run_gets_its_own_file(self, tmp_path):
        first = tmp_path / "run_one"
        second = tmp_path / "run_two"
        first.mkdir()
        second.mkdir()

        attach_run_log(str(first), "first")
        logging.getLogger("test").info("in the first run")

        attach_run_log(str(second), "second")
        logging.getLogger("test").info("in the second run")

        first_lines = (first / "run.jsonl").read_text()
        second_lines = (second / "run.jsonl").read_text()

        assert "in the first run" in first_lines
        assert "in the second run" in second_lines
        assert (
            "in the second run" not in first_lines
        ), "second run's records leaked into the first run's log"

    def test_replaces_rather_than_accumulates(self, tmp_path):
        first = tmp_path / "a"
        second = tmp_path / "b"
        first.mkdir()
        second.mkdir()

        attach_run_log(str(first), "a")
        attach_run_log(str(second), "b")

        run_logs = [
            h
            for h in logging.getLogger().handlers
            if getattr(h, "_networksynth_run_log", False)
        ]
        assert len(run_logs) == 1

    def test_skips_when_there_is_no_run_directory(self, tmp_path):
        missing = str(tmp_path / "never_created")

        assert attach_run_log(missing, "x") is None
        assert not os.path.exists(missing)

    def test_console_and_run_log_coexist(self, tmp_path):
        configure_console()
        attach_run_log(str(tmp_path), "both")

        assert len(_our_handlers()) == 2


class TestConfigNoLongerOwnsLogging:
    def test_initialize_sets_run_id_without_a_file_handler(self, tmp_path):
        from configs.generate_mode.config_sample import SampleConfig

        config = type("LogTestCfg", (SampleConfig,), {})
        config.BASE_OUTPUT_PATH = str(tmp_path)
        config.RUN_ID = ""
        config.initialize()

        assert config.RUN_ID, "initialize() must set RUN_ID"
        run_logs = [
            h
            for h in logging.getLogger().handlers
            if getattr(h, "_networksynth_run_log", False)
        ]
        assert not run_logs, "initialize() must not attach a run log"

    def test_no_latch_attribute_remains(self):
        from configs.base_config import BaseConfig

        assert not hasattr(BaseConfig, "_logger_initialized")
        assert not hasattr(BaseConfig, "_setup_logger")
