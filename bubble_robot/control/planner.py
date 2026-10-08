"""Bubble planner: где дрон должен быть относительно человека и как туда двигаться."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config

Vec2 = tuple[float, float]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Plan:
    target: Vec2  # целевая точка на кольце вокруг человека
    velocity: Vec2  # желаемая горизонтальная скорость


class BubblePlanner:
    """Работает в полярных координатах вокруг человека: радиус держится отдельно от
    перехода на выбранную сторону. Поэтому дрон обходит человека по дуге и никогда
    не срезает путь через запретный радиус."""

    FEASIBLE_STEP_DEG = 5

    def __init__(self, cfg: Config):
        self.radius = cfg.bubble.target_radius_m
        self.side = math.radians(cfg.bubble.side_deg)
        self.max_speed = cfg.safety.max_speed_mps
        self.k_r = cfg.control.radial_gain
        self.k_t = cfg.control.tangential_gain
        self.half_x, self.half_y = cfg.geofence_half_extents

    def _ring_point(self, human: Vec2, angle: float) -> Vec2:
        return human[0] + self.radius * math.cos(angle), human[1] + self.radius * math.sin(angle)

    def _inside_geofence(self, p: Vec2) -> bool:
        return abs(p[0]) <= self.half_x and abs(p[1]) <= self.half_y

    def target_angle(self, human: Vec2, heading: float) -> float:
        """Угол выбранной стороны; если точка за геозоной — ближайший допустимый угол."""
        desired = heading + self.side
        for step in range(0, 181, self.FEASIBLE_STEP_DEG):
            for sign in (1, -1):
                angle = desired + sign * math.radians(step)
                if self._inside_geofence(self._ring_point(human, angle)):
                    return angle
        return desired

    def plan(self, human: Vec2, heading: float, drone: Vec2) -> Plan:
        dx, dy = drone[0] - human[0], drone[1] - human[1]
        r = math.hypot(dx, dy)
        angle_target = self.target_angle(human, heading)
        angle = math.atan2(dy, dx) if r > 1e-6 else angle_target
        e_r = (math.cos(angle), math.sin(angle))
        e_t = (-e_r[1], e_r[0])

        # Радиус в приоритете: сначала отойти/подойти, остаток скорости — на обход.
        v_r = max(-self.max_speed, min(self.max_speed, self.k_r * (self.radius - r)))
        budget = math.sqrt(max(0.0, self.max_speed**2 - v_r**2))
        v_t = self.k_t * r * wrap_angle(angle_target - angle)
        v_t = max(-budget, min(budget, v_t))

        return Plan(
            target=self._ring_point(human, angle_target),
            velocity=(v_r * e_r[0] + v_t * e_t[0], v_r * e_r[1] + v_t * e_t[1]),
        )
