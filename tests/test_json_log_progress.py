# SPDX-License-Identifier: GPL-3.0-or-later
import json
import logging

import pytest

from networksynth.utils import JsonFormatter, progress, tagged

pytestmark = pytest.mark.unit


def _emit(message: str, extra: dict | None = None) -> dict:
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
    return json.loads(JsonFormatter(run_id="abc123").format(record))


class TestPercentField:
    def test_percent_is_promoted_to_a_top_level_field(self):
        entry = _emit("Synthetic graph generated.", tagged("PROGRESS", percent=42.5))

        assert entry["percent"] == 42.5
        assert entry["tag"] == "PROGRESS"

    def test_percent_is_numeric_not_a_string(self):
        entry = _emit("halfway", tagged("PROGRESS", percent=50.0))

        assert isinstance(entry["percent"], (int, float))
        assert not isinstance(entry["percent"], str)

    def test_absent_when_not_a_progress_record(self):
        entry = _emit("just a message", tagged("IO"))

        assert "percent" not in entry
        assert entry["tag"] == "IO"

    def test_zero_percent_is_still_emitted(self):
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
        line = JsonFormatter(run_id="r").format(
            logging.LogRecord(
                "l", logging.INFO, __file__, 1, "multi\nline\nmessage", (), None
            )
        )

        assert "\n" not in line
        assert json.loads(line)["message"] == "multi\nline\nmessage"


class TestProgressField:
    def test_it_reports_a_share_of_the_expected_total(self):
        assert progress(25, 100) == {"percent": 25.0}

    def test_it_stays_under_a_hundred_until_the_run_itself_ends(self):
        assert progress(100, 100)["percent"] == 99.0
        assert progress(500, 100)["percent"] == 99.0

    def test_no_field_when_there_is_nothing_to_measure_against(self):
        assert progress(10, 0) == {}

    def test_it_promotes_through_the_formatter(self):
        entry = _emit("Phase 2 round 9", tagged("PHASE2", **progress(1, 4)))

        assert entry["percent"] == 25.0
        assert entry["tag"] == "PHASE2"


class TestPipelinesEmitIt:

    # Either idiom carries the field: a literal percent=, or the shared helper
    # splatted into tagged(). Grepping the source is what catches a new site
    # that logs a percentage and forgets one.
    _IDIOMS = ("percent=", "**progress(")

    @pytest.mark.parametrize(
        "module_name",
        ["generate", "sweep", "hybrid"],
    )
    def test_progress_sites_pass_percent(self, module_name):
        import inspect
        import re

        module = __import__(f"networksynth.pipelines.{module_name}", fromlist=["x"])
        source = inspect.getsource(module)

        for match in re.finditer(r"logger\.info\((.{0,400}?)\)\n", source, re.S):
            body = match.group(1)
            if "%)" in body or "%" in body and "progress" in body:
                assert any(idiom in body for idiom in self._IDIOMS), (
                    f"networksynth.pipelines.{module_name} logs a percentage without the "
                    f"structured field:\n{body.strip()[:200]}"
                )
