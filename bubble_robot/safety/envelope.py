"""Ограничение setpoints и контроль нарушений safety envelope."""

from __future__ import annotations

import math

from ..config import Config
from .state_machine import State

Vec3 = tuple[float, float, float]


def _clamp_norm(v: Vec3, limit: float) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n <= limit or n == 0.0:
        return v
    k = limit / n
    return v[0] * k, v[1] * k, v[2] * k


class SetpointLimiter:
    """Последний рубеж перед дроном: что бы ни попросил планировщик, наружу уходит
    velocity setpoint в пределах скорости, ускорения, геозоны и потолка высоты."""

    def __init__(self, cfg: Config):
        self.max_speed = cfg.safety.max_speed_mps
        self.max_accel = cfg.safety.max_accel_mps2
        self.altitude_max = cfg.safety.altitude_max_m
        self.half_x, self.half_y = cfg.geofence_half_extents

    def limit(self, v_cmd: Vec3, v_prev: Vec3, pos: Vec3, dt: float) -> Vec3:
        v = _clamp_norm(v_cmd, self.max_speed)
        dv = _clamp_norm((v[0] - v_prev[0], v[1] - v_prev[1], v[2] - v_prev[2]), self.max_accel * dt)
        vx, vy, vz = v_prev[0] + dv[0], v_prev[1] + dv[1], v_prev[2] + dv[2]
        # Границы важнее плавности: наружу за геозону и выше потолка — ноль сразу.
        if (pos[0] >= self.half_x and vx > 0) or (pos[0] <= -self.half_x and vx < 0):
            vx = 0.0
        if (pos[1] >= self.half_y and vy > 0) or (pos[1] <= -self.half_y and vy < 0):
            vy = 0.0
        if pos[2] >= self.altitude_max and vz > 0:
            vz = 0.0
        return vx, vy, vz


class EnvelopeMonitor:
    """Проверяет фактическое состояние, а не команды. Возвращает коды нарушений."""

    SPEED_TOLERANCE = 1.02

    def __init__(self, cfg: Config):
        self.safety = cfg.safety
        self.half_room = (cfg.room.width_m / 2, cfg.room.depth_m / 2)

    def check(self, state: State, pos: Vec3, vel: Vec3, human_distance_m: float) -> list[str]:
        if state in (State.DISARMED, State.EMERGENCY_STOP):
            return []
        s = self.safety
        violations = []
        if human_distance_m < s.min_human_distance_m:
            violations.append("min_human_distance")
        if math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2) > s.max_speed_mps * self.SPEED_TOLERANCE:
            violations.append("max_speed")
        if pos[2] > s.altitude_max_m:
            violations.append("altitude_max")
        if state in (State.ORBIT, State.HOLD) and pos[2] < s.altitude_min_m:
            violations.append("altitude_min")
        if abs(pos[0]) > self.half_room[0] or abs(pos[1]) > self.half_room[1]:
            violations.append("outside_room")
        return violations
