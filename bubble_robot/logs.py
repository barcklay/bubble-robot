"""Лог прогона: телеметрия в CSV (строка на такт) и события в JSONL."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

TELEMETRY_FIELDS = [
    "t",
    "state",
    "human_x",
    "human_y",
    "human_heading",
    "drone_x",
    "drone_y",
    "drone_z",
    "drone_vx",
    "drone_vy",
    "drone_vz",
    "target_x",
    "target_y",
    "distance",
    "speed",
    "tracking_valid",
    "tracking_age",
    "battery",
    "violations",
]


class RunLogger:
    def __init__(self, log_dir: str | Path, name: str | None = None):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        name = name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.telemetry_path = log_dir / f"{name}.csv"
        self.events_path = log_dir / f"{name}.events.jsonl"
        self._telemetry_file = open(self.telemetry_path, "w", newline="", encoding="utf-8")
        self._events_file = open(self.events_path, "w", encoding="utf-8")
        self._writer = csv.DictWriter(self._telemetry_file, fieldnames=TELEMETRY_FIELDS)
        self._writer.writeheader()

    def telemetry(self, row: dict) -> None:
        self._writer.writerow(row)

    def event(self, t: float, kind: str, **data) -> None:
        self._events_file.write(json.dumps({"t": round(t, 3), "kind": kind, **data}, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._telemetry_file.close()
        self._events_file.close()
