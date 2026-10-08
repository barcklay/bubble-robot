"""Скриптовый демо-сценарий: то же, что HW-94 требует от живого полёта, но в симуляторе."""

from __future__ import annotations

import math

from ..safety.state_machine import Command
from .world import Simulation

WAYPOINTS = [(1.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (0.5, -1.0), (0.0, 0.0)]
WAYPOINT_PAUSE_S = 3.0
WAYPOINT_REACHED_M = 0.05


class DemoScenario:
    """Человек медленно ходит по точкам; по ходу — короткая потеря трекинга (→ Hold и
    обратно), ручной Hold/Resume и штатная посадка в конце."""

    def __init__(self, duration_s: float = 300.0):
        self.duration_s = duration_s
        land_at = duration_s - 15.0
        mid_flight = [(60.0, "dropout_on"), (60.8, "dropout_off"), (120.0, "hold"), (128.0, "resume")]
        # В коротком прогоне события, не успевающие до посадки, просто не происходят.
        self._events = [(1.0, "takeoff"), *[e for e in mid_flight if e[0] < land_at - 10.0], (land_at, "land")]
        self._waypoint = 0
        self._pause_until = 0.0
        self._human_frozen = False

    def apply(self, sim: Simulation) -> None:
        """Вызывать перед каждым sim.step()."""
        while self._events and sim.t >= self._events[0][0]:
            self._fire(self._events.pop(0)[1], sim)
        sim.set_human_velocity(*self._human_velocity(sim))

    def _fire(self, action: str, sim: Simulation) -> None:
        if action == "takeoff":
            sim.command(Command.TAKEOFF)
        elif action == "dropout_on":
            sim.tracker.dropout = True
        elif action == "dropout_off":
            sim.tracker.dropout = False
        elif action == "hold":
            # Правило испытаний: пока дрон в ручном Hold, человек к нему не идёт.
            sim.command(Command.HOLD)
            self._human_frozen = True
        elif action == "resume":
            sim.command(Command.RESUME)
            self._human_frozen = False
        elif action == "land":
            sim.command(Command.LAND)
            self._human_frozen = True

    def _human_velocity(self, sim: Simulation) -> tuple[float, float]:
        if self._human_frozen or sim.t < self._pause_until:
            return 0.0, 0.0
        wx, wy = WAYPOINTS[self._waypoint]
        dx, dy = wx - sim.human[0], wy - sim.human[1]
        dist = math.hypot(dx, dy)
        if dist < WAYPOINT_REACHED_M:
            self._waypoint = (self._waypoint + 1) % len(WAYPOINTS)
            self._pause_until = sim.t + WAYPOINT_PAUSE_S
            return 0.0, 0.0
        speed = sim.cfg.sim.human_speed_mps
        return dx / dist * speed, dy / dist * speed


def run_scenario(sim: Simulation, scenario: DemoScenario) -> None:
    steps = round(scenario.duration_s / sim.dt)
    for _ in range(steps):
        scenario.apply(sim)
        sim.step()
