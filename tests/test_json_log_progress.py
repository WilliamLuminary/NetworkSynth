# tests/test_json_log_progress.py
"""
The JSON log must carry a machine-readable ``percent`` field.

StructuralGT's controller is meant to tail this file and forward each line as a
``ProgressData``, which wants a numeric percent.  Percentages used to live only
inside the message text — ``"(50.0%) Synthetic graph generated."`` — which would
have forced the caller to parse prose.  See ``INTEGRATION_PLAN.md``.
"""

import json
import logging

import pytest

from configs.base_config import _JsonFormatter, tagged

pytestmark = pytest.mark.unit


def _emit(message: str, extra: dict | None = None) -> dict:
    """Format one record through the JSON formatter and return the parsed line."""
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return json.loads(_JsonFormatter(run_id="abc123").format(record))


class TestPercentField:
    def test_percent_is_promoted_to_a_top_level_field(self):
        entry = _emit("Synthetic graph generated.", tagged("PROGRESS", percent=42.5))

        assert entry["percent"] == 42.5
        assert entry["tag"] == "PROGRESS"

    def test_percent_is_numeric_not_a_string(self):
        """A consumer must be able to use it without coercion."""
        entry = _emit("halfway", tagged("PROGRESS", percent=50.0))

        assert isinstance(entry["percent"], (int, float))
        assert not isinstance(entry["percent"], str)

    def test_absent_when_not_a_progress_record(self):
        entry = _emit("just a message", tagged("IO"))

        assert "percent" not in entry
        assert entry["tag"] == "IO"

    def test_zero_percent_is_still_emitted(self):
        """0 is falsy; it must not be dropped."""
        entry = _emit("starting", tagged("PROGRESS", percent=0))

        assert entry["percent"] == 0

    def test_keeps_the_standard_fields(self):
        entry = _emit("hello", tagged("PROGRESS", percent=10, dataset="sample_A"))

        assert entry["run_id"] == "abc123"
        assert entry["level"] == "INFO"
        assert entry["logger"] == "test.logger"
        assert entry["message"] == "hello"
        assert entry["dataset"] == "sample_A"

    def test_line_is_a_single_json_object(self):
        """One record per line, so a tailing consumer can parse incrementally."""
        line = _JsonFormatter(run_id="r").format(
            logging.LogRecord(
                "l", logging.INFO, __file__, 1, "multi\nline\nmessage", (), None
            )
        )

        assert "\n" not in line
        assert json.loads(line)["message"] == "multi\nline\nmessage"


class TestPipelinesEmitIt:
    """Every progress log site must attach the field, not just format it."""

    @pytest.mark.parametrize(
        "module_name",
        ["generate", "sweep", "generate_from_props", "mosaic", "hybrid"],
    )
    def test_progress_sites_pass_percent(self, module_name):
        import inspect
        import re

        module = __import__(f"pipelines.{module_name}", fromlist=["x"])
        source = inspect.getsource(module)

        # every logger call mentioning a percent in its message must also carry
        # the structured field
        for match in re.finditer(r"logger\.info\((.{0,400}?)\)\n", source, re.S):
            body = match.group(1)
            if "%)" in body or "%" in body and "progress" in body:
                assert "percent=" in body, (
                    f"pipelines.{module_name} logs a percentage without the "
                    f"structured field:\n{body.strip()[:200]}"
                )
