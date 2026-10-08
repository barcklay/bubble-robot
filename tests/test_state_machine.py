import itertools

import pytest

from bubble_robot.safety.state_machine import ALLOWED, Command, Inputs, SafetyStateMachine, State

OK = Inputs(tracking_age_s=0.05, altitude_m=0.0, battery=1.0, human_distance_m=2.0)
FLYING = Inputs(tracking_age_s=0.05, altitude_m=0.8, battery=1.0, human_distance_m=1.75)


def lost(age: float) -> Inputs:
    return Inputs(tracking_age_s=age, altitude_m=0.8, battery=1.0, human_distance_m=1.75)


@pytest.fixture
def sm(cfg):
    return SafetyStateMachine(cfg.safety, cfg.bubble.flight_altitude_m)


@pytest.fixture
def orbiting(sm):
    assert sm.command(Command.TAKEOFF, 0.0, OK)
    assert sm.update(1.0, FLYING) is State.ORBIT
    return sm


def test_normal_flight_cycle(sm):
    assert sm.state is State.DISARMED
    assert sm.command(Command.TAKEOFF, 0.0, OK)
    assert sm.update(0.5, Inputs(0.05, 0.4, 1.0, 2.0)) is State.TAKEOFF
    assert sm.update(1.0, FLYING) is State.ORBIT
    assert sm.command(Command.LAND, 2.0, FLYING)
    assert sm.update(2.5, Inputs(0.05, 0.3, 1.0, 1.75)) is State.LAND
    assert sm.update(3.0, Inputs(0.05, 0.0, 1.0, 1.75)) is State.DISARMED


# --- запрещённые переходы ---------------------------------------------------


def test_every_transition_outside_table_is_blocked(sm):
    for source, target in itertools.product(State, State):
        sm.state = source
        allowed = target in ALLOWED[source]
        assert sm._go(target, 0.0, "тест") is allowed
        assert sm.state is (target if allowed else source)


@pytest.mark.parametrize("cmd", [Command.HOLD, Command.RESUME, Command.LAND, Command.RESET])
def test_flight_commands_are_rejected_when_disarmed(sm, cmd):
    assert not sm.command(cmd, 0.0, OK)
    assert sm.state is State.DISARMED
    assert sm.rejected


def test_takeoff_is_rejected_in_flight(orbiting):
    assert not orbiting.command(Command.TAKEOFF, 2.0, FLYING)
    assert orbiting.state is State.ORBIT


def test_landing_cannot_be_cancelled(orbiting):
    orbiting.command(Command.LAND, 2.0, FLYING)
    for cmd in (Command.RESUME, Command.HOLD, Command.TAKEOFF):
        assert not orbiting.command(cmd, 2.1, FLYING)
    assert orbiting.update(2.2, FLYING) is State.LAND


@pytest.mark.parametrize(
    "inputs",
    [
        Inputs(tracking_age_s=float("inf"), battery=1.0, human_distance_m=2.0),
        Inputs(tracking_age_s=0.05, battery=0.15, human_distance_m=2.0),
        Inputs(tracking_age_s=0.05, battery=1.0, human_distance_m=1.2),
    ],
    ids=["нет трекинга", "низкий заряд", "человек слишком близко"],
)
def test_takeoff_preconditions(sm, inputs):
    assert not sm.command(Command.TAKEOFF, 0.0, inputs)
    assert sm.state is State.DISARMED


# --- потеря трекера ---------------------------------------------------------


def test_tracking_loss_holds_then_lands(orbiting):
    assert orbiting.update(2.0, lost(0.25)) is State.ORBIT
    assert orbiting.update(2.1, lost(0.35)) is State.HOLD
    assert orbiting.update(3.0, lost(1.9)) is State.HOLD
    assert orbiting.update(3.2, lost(2.1)) is State.LAND


def test_tracking_recovery_resumes_orbit(orbiting):
    assert orbiting.update(2.0, lost(0.5)) is State.HOLD
    assert orbiting.update(2.1, FLYING) is State.ORBIT


def test_manual_hold_is_not_resumed_automatically(orbiting):
    assert orbiting.command(Command.HOLD, 2.0, FLYING)
    assert orbiting.update(2.1, FLYING) is State.HOLD
    assert orbiting.command(Command.RESUME, 3.0, FLYING)
    assert orbiting.state is State.ORBIT


def test_resume_without_tracking_stays_in_hold(orbiting):
    orbiting.command(Command.HOLD, 2.0, FLYING)
    assert not orbiting.command(Command.RESUME, 2.5, lost(1.0))
    assert orbiting.state is State.HOLD
    # Ручной Hold снят, поэтому с возвратом трекинга дрон продолжит сам.
    assert orbiting.update(2.6, FLYING) is State.ORBIT


def test_tracking_lost_during_takeoff(sm):
    sm.command(Command.TAKEOFF, 0.0, OK)
    assert sm.update(1.0, lost(0.5)) is State.HOLD
    sm2 = SafetyStateMachine(sm.safety, sm.flight_altitude_m)
    sm2.command(Command.TAKEOFF, 0.0, OK)
    assert sm2.update(1.0, Inputs(2.5, 0.3, 1.0, 2.0)) is State.LAND


def test_hold_requested_during_takeoff_applies_at_altitude(sm):
    sm.command(Command.TAKEOFF, 0.0, OK)
    assert sm.command(Command.HOLD, 0.5, Inputs(0.05, 0.3, 1.0, 2.0))
    assert sm.state is State.TAKEOFF
    assert sm.update(1.0, FLYING) is State.HOLD
    assert sm.update(1.5, FLYING) is State.HOLD


def test_low_battery_lands(orbiting):
    assert orbiting.update(2.0, Inputs(0.05, 0.8, 0.19, 1.75)) is State.LAND


# --- emergency stop ---------------------------------------------------------


@pytest.mark.parametrize("state", list(State))
def test_emergency_stop_from_any_state(sm, state):
    sm.state = state
    assert sm.command(Command.EMERGENCY_STOP, 0.0, lost(5.0))
    assert sm.state is State.EMERGENCY_STOP


def test_emergency_stop_latches_until_reset(orbiting):
    orbiting.command(Command.EMERGENCY_STOP, 2.0, FLYING)
    for cmd in (Command.TAKEOFF, Command.RESUME, Command.HOLD, Command.LAND):
        assert not orbiting.command(cmd, 2.1, OK)
    assert orbiting.update(2.2, FLYING) is State.EMERGENCY_STOP
    assert orbiting.command(Command.RESET, 3.0, OK)
    assert orbiting.state is State.DISARMED


def test_emergency_stop_beats_automatic_transitions(orbiting):
    orbiting.command(Command.EMERGENCY_STOP, 2.0, FLYING)
    # Ни потеря трекинга, ни низкий заряд не выводят из аварийной остановки.
    assert orbiting.update(2.1, Inputs(9.0, 0.5, 0.05, 1.0)) is State.EMERGENCY_STOP
