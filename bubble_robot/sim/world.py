"""Симуляция: человек, трекер, дрон и контур управления в одном такте.

Один и тот же Simulation крутится и в окне (ui/sim_app.py), и без графики
(headless-прогон сценария и тесты).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from ..config import Config
from ..control.planner import BubblePlanner
from ..drone.simulated import SimulatedDrone
from ..logs import RunLogger
from ..safety.envelope import EnvelopeMonitor, SetpointLimiter
from ..safety.state_machine import AIRBORNE, Command, Inputs, SafetyStateMachine, State
from ..tracking.base import TrackingSample
from ..tracking.simulated import SimulatedTracker

HUMAN_WALL_MARGIN_M = 0.2
HEADING_MIN_SPEED_MPS = 0.05


@dataclass
class SimStats:
    min_distance_m: float = math.inf
    max_speed_mps: float = 0.0
    violation_ticks: Counter = field(default_factory=Counter)
    state_ticks: Counter = field(default_factory=Counter)
    orbit_in_band_ticks: int = 0

    @property
    def violations(self) -> int:
        return sum(self.violation_ticks.values())


class Simulation:
    def __init__(self, cfg: Config, logger: RunLogger | None = None, seed: int = 0):
        self.cfg = cfg
        self.dt = 1.0 / cfg.control.rate_hz
        self.t = 0.0
        self.logger = logger

        self.human = [0.0, 0.0]
        self.human_vel = [0.0, 0.0]
        self.human_heading = 0.0
        self._human_cmd = (0.0, 0.0)

        self.tracker = SimulatedTracker(cfg.sim, random.Random(seed))
        self.drone = SimulatedDrone(cfg.sim)
        self.sm = SafetyStateMachine(cfg.safety, cfg.bubble.flight_altitude_m)
        self.planner = BubblePlanner(cfg)
        self.limiter = SetpointLimiter(cfg)
        self.monitor = EnvelopeMonitor(cfg)

        self.last_valid: TrackingSample | None = None
        self.target: tuple[float, float] | None = None
        self.distance = self._true_distance()
        self.stats = SimStats()
        self._inputs = Inputs()
        self._setpoint = (0.0, 0.0, 0.0)
        self._logged_transitions = 0
        self._logged_rejected = 0
        self._active_violations: set[str] = set()

    # --- управление извне -------------------------------------------------

    def set_human_velocity(self, vx: float, vy: float) -> None:
        speed = math.hypot(vx, vy)
        limit = self.cfg.sim.human_speed_mps
        if speed > limit:
            vx, vy = vx * limit / speed, vy * limit / speed
        self._human_cmd = (vx, vy)

    def command(self, cmd: Command) -> bool:
        accepted = self.sm.command(cmd, self.t, self._inputs)
        self._apply_motor_state()
        self._flush_events()
        return accepted

    @property
    def state(self) -> State:
        return self.sm.state

    @property
    def tracking_age(self) -> float:
        return self._inputs.tracking_age_s

    # --- такт ---------------------------------------------------------------

    def step(self) -> None:
        dt = self.dt
        self.t += dt
        self._step_human(dt)

        self.tracker.observe(self.t, self.human[0], self.human[1], self.human_heading)
        sample = self.tracker.latest(self.t)
        if sample is not None and sample.tracking_valid:
            self.last_valid = sample

        pos = tuple(self.drone.pos)
        tracked = self.last_valid
        self._inputs = Inputs(
            tracking_age_s=self.t - tracked.timestamp if tracked else math.inf,
            altitude_m=pos[2],
            battery=self.drone.battery,
            human_distance_m=math.hypot(pos[0] - tracked.x, pos[1] - tracked.y) if tracked else math.inf,
        )
        state = self.sm.update(self.t, self._inputs)
        self._apply_motor_state()

        if state in AIRBORNE:
            self._setpoint = self.limiter.limit(self._desired_velocity(state, pos), self._setpoint, pos, dt)
            self.drone.send_velocity_setpoint(self._setpoint)
        else:
            self._setpoint = (0.0, 0.0, 0.0)
            self.target = None
        self.drone.step(dt)

        self._record(state)

    def _step_human(self, dt: float) -> None:
        max_dv = self.cfg.sim.human_accel_mps2 * dt
        for i in range(2):
            dv = self._human_cmd[i] - self.human_vel[i]
            self.human_vel[i] += max(-max_dv, min(max_dv, dv))
        half = (
            self.cfg.room.width_m / 2 - HUMAN_WALL_MARGIN_M,
            self.cfg.room.depth_m / 2 - HUMAN_WALL_MARGIN_M,
        )
        for i in range(2):
            self.human[i] = max(-half[i], min(half[i], self.human[i] + self.human_vel[i] * dt))
        if math.hypot(*self.human_vel) > HEADING_MIN_SPEED_MPS:
            self.human_heading = math.atan2(self.human_vel[1], self.human_vel[0])

    def _desired_velocity(self, state: State, pos: tuple) -> tuple[float, float, float]:
        control = self.cfg.control
        climb = control.altitude_gain * (self.cfg.bubble.flight_altitude_m - pos[2])
        self.target = None
        if state is State.LAND:
            return 0.0, 0.0, -control.land_speed_mps
        if state is State.ORBIT and self.last_valid is not None:
            tracked = self.last_valid
            plan = self.planner.plan((tracked.x, tracked.y), tracked.heading, (pos[0], pos[1]))
            self.target = plan.target
            return plan.velocity[0], plan.velocity[1], climb
        # TAKEOFF и HOLD: по горизонтали стоим на месте.
        return 0.0, 0.0, climb

    def _apply_motor_state(self) -> None:
        if self.sm.state in AIRBORNE:
            if not self.drone.motors_on:
                self.drone.start_motors()
        elif self.drone.motors_on:
            self.drone.stop_motors()

    # --- учёт и лог ---------------------------------------------------------

    def _true_distance(self) -> float:
        return math.hypot(self.drone.pos[0] - self.human[0], self.drone.pos[1] - self.human[1])

    def _record(self, state: State) -> None:
        pos, vel = tuple(self.drone.pos), tuple(self.drone.vel)
        # Нарушения считаются по истинному положению человека, а не по данным трекера.
        self.distance = self._true_distance()
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        violations = self.monitor.check(state, pos, vel, self.distance)

        stats = self.stats
        stats.state_ticks[state] += 1
        if state in AIRBORNE:
            stats.min_distance_m = min(stats.min_distance_m, self.distance)
            stats.max_speed_mps = max(stats.max_speed_mps, speed)
        if state is State.ORBIT:
            bubble = self.cfg.bubble
            stats.orbit_in_band_ticks += bubble.radius_min_m <= self.distance <= bubble.radius_max_m
        for code in violations:
            stats.violation_ticks[code] += 1

        if self.logger is None:
            return
        for code in set(violations) - self._active_violations:
            self.logger.event(self.t, "violation", code=code, distance=round(self.distance, 3), speed=round(speed, 3))
        self._active_violations = set(violations)
        self._flush_events()
        sample = self.tracker.latest(self.t)
        self.logger.telemetry(
            {
                "t": round(self.t, 3),
                "state": state.value,
                "human_x": round(self.human[0], 4),
                "human_y": round(self.human[1], 4),
                "human_heading": round(self.human_heading, 4),
                "drone_x": round(pos[0], 4),
                "drone_y": round(pos[1], 4),
                "drone_z": round(pos[2], 4),
                "drone_vx": round(vel[0], 4),
                "drone_vy": round(vel[1], 4),
                "drone_vz": round(vel[2], 4),
                "target_x": round(self.target[0], 4) if self.target else "",
                "target_y": round(self.target[1], 4) if self.target else "",
                "distance": round(self.distance, 4),
                "speed": round(speed, 4),
                "tracking_valid": int(bool(sample and sample.tracking_valid)),
                "tracking_age": round(self.tracking_age, 3) if math.isfinite(self.tracking_age) else "",
                "battery": round(self.drone.battery, 4),
                "violations": "|".join(violations),
            }
        )

    def _flush_events(self) -> None:
        if self.logger is None:
            return
        for tr in self.sm.history[self._logged_transitions :]:
            self.logger.event(tr.t, "transition", source=tr.source.value, target=tr.target.value, reason=tr.reason)
        for tr in self.sm.rejected[self._logged_rejected :]:
            self.logger.event(tr.t, "rejected", source=tr.source.value, target=tr.target.value, reason=tr.reason)
        self._logged_transitions = len(self.sm.history)
        self._logged_rejected = len(self.sm.rejected)
