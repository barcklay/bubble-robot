"""Симулированный трекер: частота, задержка, шум и управляемая потеря цели."""

from __future__ import annotations

import random
from collections import deque

from ..config import SimConfig
from .base import TrackingSample


class SimulatedTracker:
    def __init__(self, sim: SimConfig, rng: random.Random):
        self.period = 1.0 / sim.tracker_rate_hz
        self.latency = sim.tracker_latency_s
        self.noise = sim.tracker_noise_m
        self.rng = rng
        self.dropout = False  # True — метка «потеряна», сэмплы идут с tracking_valid=False
        self._next_capture = 0.0
        self._in_flight: deque[tuple[float, TrackingSample]] = deque()
        self._latest: TrackingSample | None = None

    def observe(self, t: float, x: float, y: float, heading: float) -> None:
        """Сообщить трекеру истинное положение человека на момент t."""
        if t + 1e-9 < self._next_capture:
            return
        self._next_capture += self.period
        valid = not self.dropout
        sample = TrackingSample(
            timestamp=t,
            x=x + self.rng.gauss(0, self.noise),
            y=y + self.rng.gauss(0, self.noise),
            heading=heading,
            confidence=1.0 if valid else 0.0,
            tracking_valid=valid,
        )
        self._in_flight.append((t + self.latency, sample))

    def latest(self, t: float) -> TrackingSample | None:
        """Последний сэмпл, который к моменту t уже доехал до приложения."""
        while self._in_flight and self._in_flight[0][0] <= t + 1e-9:
            self._latest = self._in_flight.popleft()[1]
        return self._latest
