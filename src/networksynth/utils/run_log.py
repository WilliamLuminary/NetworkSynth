# SPDX-License-Identifier: GPL-3.0-or-later
import json
import logging


def progress(made: int, expected: int) -> dict:
    """A share of the nodes a run expects to make, as a log extra.

    Node count is the one quantity that tracks elapsed time, and its endpoint is
    predictable because an assembled network fills its frame at about the density
    of the network it was modelled on. Rounds are no use: a measured run
    converged at round 204 of a 500 ceiling.

    Held under 100 until the run itself ends, since a bar that arrives early is
    the thing this replaces.
    """
    if expected <= 0:
        return {}
    return {"percent": round(min(made / expected * 100, 99.0), 1)}


def tagged(tag: str, **extra) -> dict:
    return {"tag": tag, **extra}


class JsonFormatter(logging.Formatter):

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
