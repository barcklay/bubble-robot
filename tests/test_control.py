import math

import pytest

from bubble_robot.control.planner import BubblePlanner
from bubble_robot.safety.envelope import EnvelopeMonitor, SetpointLimiter
from bubble_robot.safety.state_machine import State

DT = 0.02


def norm(v):
    return math.sqrt(sum(c * c for c in v))


# --- ограничитель setpoints -------------------------------------------------


def test_speed_is_clamped(cfg):
    limiter = SetpointLimiter(cfg)
    v = (0.0, 0.0, 0.0)
    for _ in range(500):
        v = limiter.limit((5.0, -3.0, 1.0), v, (0.0, 0.0, 0.8), DT)
        assert norm(v) <= cfg.safety.max_speed_mps + 1e-9
    assert norm(v) == pytest.approx(cfg.safety.max_speed_mps)


def test_acceleration_is_clamped(cfg):
    limiter = SetpointLimiter(cfg)
    v = limiter.limit((0.3, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.8), DT)
    assert norm(v) == pytest.approx(cfg.safety.max_accel_mps2 * DT)


def test_geofence_and_ceiling_stop_outward_motion(cfg):
    limiter = SetpointLimiter(cfg)
    half_x, half_y = cfg.geofence_half_extents
    moving = (0.2, 0.2, 0.2)
    v = limiter.limit(moving, moving, (half_x, -half_y, cfg.safety.altitude_max_m), DT)
    assert v[0] == 0.0  # наружу по x — нельзя
    assert v[1] > 0.0  # внутрь по y — можно
    assert v[2] == 0.0  # выше потолка — нельзя


def test_monitor_flags_each_limit(cfg):
    monitor = EnvelopeMonitor(cfg)
    still = (0.0, 0.0, 0.0)
    assert monitor.check(State.ORBIT, (0.0, 0.0, 0.8), still, 1.75) == []
    assert monitor.check(State.ORBIT, (0.0, 0.0, 0.8), still, 1.4) == ["min_human_distance"]
    assert monitor.check(State.ORBIT, (0.0, 0.0, 0.8), (0.4, 0.0, 0.0), 1.75) == ["max_speed"]
    assert monitor.check(State.ORBIT, (0.0, 0.0, 1.3), still, 1.75) == ["altitude_max"]
    assert monitor.check(State.HOLD, (0.0, 0.0, 0.3), still, 1.75) == ["altitude_min"]
    assert monitor.check(State.TAKEOFF, (0.0, 0.0, 0.3), still, 1.75) == []
    assert monitor.check(State.DISARMED, (0.0, 0.0, 0.0), still, 0.5) == []


# --- планировщик ------------------------------------------------------------


def test_target_is_on_chosen_side_at_target_radius(cfg):
    planner = BubblePlanner(cfg)
    plan = planner.plan((0.0, 0.0), 0.0, (0.0, 1.75))
    # Человек смотрит вдоль +x, side_deg=90 → цель слева, то есть на +y.
    assert plan.target == pytest.approx((0.0, 1.75), abs=1e-9)
    assert norm(plan.velocity) == pytest.approx(0.0, abs=1e-9)


def test_retreats_straight_back_when_too_close(cfg):
    planner = BubblePlanner(cfg)
    vx, vy = planner.plan((0.0, 0.0), 0.0, (1.0, 0.0)).velocity
    assert vx == pytest.approx(cfg.safety.max_speed_mps)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_goes_around_not_through_the_human(cfg):
    planner = BubblePlanner(cfg)
    # Дрон справа от человека, цель слева: кратчайший путь лежит через человека.
    human, drone = (0.0, 0.0), [0.0, -1.75]
    min_r = math.inf
    for _ in range(4000):
        vx, vy = planner.plan(human, 0.0, tuple(drone)).velocity
        assert norm((vx, vy)) <= cfg.safety.max_speed_mps + 1e-9
        drone[0] += vx * DT
        drone[1] += vy * DT
        min_r = min(min_r, math.hypot(*drone))
    assert min_r >= cfg.safety.min_human_distance_m
    assert drone == pytest.approx([0.0, 1.75], abs=0.02)


def test_target_slides_along_ring_when_side_is_outside_geofence(cfg):
    planner = BubblePlanner(cfg)
    half_x, half_y = cfg.geofence_half_extents
    human = (0.0, half_y - 0.5)  # слева от человека — стена
    tx, ty = planner.plan(human, 0.0, (1.75, human[1])).target
    assert abs(tx) <= half_x and abs(ty) <= half_y
    assert math.hypot(tx - human[0], ty - human[1]) == pytest.approx(cfg.bubble.target_radius_m)
