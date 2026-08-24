"""The shape of a log record: how a line is tagged, and how it is written.

Lives here, at the bottom of the import graph, because everything logs —
``graphs`` and ``utils`` included, and both are imported by ``handlers``, so the
pair cannot live up there without a cycle.  Setting logging *up* is a run
concern and stays in :mod:`handlers.run_logging`.
"""

import json
import logging


def tagged(tag: str, **extra) -> dict:
    """Return an ``extra`` dict for use with ``logger.info(msg, extra=tagged("PHASE1"))``.

    Usage::

        logger.info("Phase 1 complete", extra=tagged("PHASE1", dataset="sample_A"))

    The tag and any additional keys are embedded in the JSON log output
    and ignored by the plain-text console formatter.
    """
    return {"tag": tag, **extra}


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Recognised ``extra`` keys (``tag``, ``dataset``, ``percent``) are promoted
    to top-level fields so they can be filtered with ``jq``.

    ``percent`` is a machine-readable completion figure, present on progress
    records.  It exists so a caller tailing this file can drive a progress bar
    without parsing percentages out of the message text — see
    ``INTEGRATION_PLAN.md``.  Emit it with
    ``logger.info(msg, extra=tagged("PROGRESS", percent=pct))``.

    Each entry includes a ``run_id`` so concurrent runs appending to the
    same file can be distinguished: ``jq 'select(.run_id == "abc123")'``.
    """

    def __init__(self, run_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.default_time_format),
            "run_id": self.run_id,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("tag", "dataset", "percent"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)
