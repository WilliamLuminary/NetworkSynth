# SPDX-License-Identifier: GPL-3.0-or-later
import json
import logging


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
