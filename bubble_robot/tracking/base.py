"""Общий формат данных трекера — тот же, что будет отдавать сервис на Raspberry Pi (HW-87)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingSample:
    timestamp: float  # момент съёмки кадра, не момент доставки
    x: float
    y: float
    heading: float  # рад, направление человека в плоскости пола
    confidence: float
    tracking_valid: bool
