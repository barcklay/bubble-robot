import csv
import json

import pytest

from bubble_robot.logs import RunLogger
from bubble_robot.safety.state_machine import Command, State
from bubble_robot.sim.scenario import DemoScenario, run_scenario
from bubble_robot.sim.world import Simulation


def run_for(sim: Simulation, seconds: float) -> None:
    for _ in range(round(seconds / sim.dt)):
        sim.step()


@pytest.fixture
def flying(cfg):
    sim = Simulation(cfg)
    run_for(sim, 0.5)
    assert sim.command(Command.TAKEOFF)
    run_for(sim, 6.0)
    assert sim.state is State.ORBIT
    return sim


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_five_minute_scenario_stays_inside_envelope(cfg, seed):
    sim = Simulation(cfg, seed=seed)
    run_scenario(sim, DemoScenario(300.0))

    stats = sim.stats
    assert stats.violations == 0, dict(stats.violation_ticks)
    assert stats.min_distance_m >= cfg.safety.min_human_distance_m
    assert stats.max_speed_mps <= cfg.safety.max_speed_mps * 1.02
    assert stats.state_ticks[State.ORBIT] * sim.dt > 240
    assert stats.orbit_in_band_ticks / stats.state_ticks[State.ORBIT] > 0.95
    # Сценарий прошёл через Hold (потеря трекинга и ручной) и закончился на земле.
    visited = [tr.target for tr in sim.sm.history]
    assert visited.count(State.HOLD) == 2
    assert sim.state is State.DISARMED
    assert sim.drone.pos[2] == 0.0


def test_tracker_loss_holds_then_lands(flying):
    flying.tracker.dropout = True
    run_for(flying, 1.0)
    assert flying.state is State.HOLD
    run_for(flying, 1.5)
    assert flying.state is State.LAND
    run_for(flying, 6.0)
    assert flying.state is State.DISARMED
    assert flying.stats.violations == 0


def test_drone_does_not_chase_stale_target_in_hold(flying):
    flying.tracker.dropout = True
    run_for(flying, 1.5)
    assert flying.state is State.HOLD
    x, y = flying.drone.pos[:2]
    flying.set_human_velocity(-0.2, 0.0)
    run_for(flying, 0.4)
    assert flying.drone.pos[:2] == pytest.approx([x, y], abs=0.02)


def test_emergency_stop_cuts_motors_immediately(flying):
    flying.command(Command.EMERGENCY_STOP)
    assert not flying.drone.motors_on
    run_for(flying, 1.0)
    assert flying.drone.pos[2] == 0.0
    assert not flying.command(Command.TAKEOFF)
    assert flying.state is State.EMERGENCY_STOP


def test_run_is_logged(cfg, tmp_path):
    logger = RunLogger(tmp_path, name="run")
    sim = Simulation(cfg, logger)
    run_scenario(sim, DemoScenario(30.0))
    logger.close()

    rows = list(csv.DictReader(logger.telemetry_path.open(encoding="utf-8")))
    assert len(rows) == round(30.0 / sim.dt)
    orbit = [r for r in rows if r["state"] == "ORBIT"]
    assert orbit and all(r["distance"] and r["drone_x"] and r["target_x"] for r in orbit)

    events = [json.loads(line) for line in logger.events_path.open(encoding="utf-8")]
    assert [e["target"] for e in events if e["kind"] == "transition"] == ["TAKEOFF", "ORBIT", "LAND", "DISARMED"]
