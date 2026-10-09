import pygame
import pytest

from bubble_robot.safety.state_machine import State
from bubble_robot.sim.world import Simulation


@pytest.fixture
def app(cfg, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from bubble_robot.ui.sim_app import SimApp

    app = SimApp(Simulation(cfg))
    yield app
    pygame.quit()


def run_for(app, seconds: float) -> None:
    for _ in range(round(seconds / app.sim.dt)):
        app.sim.step()
    app.draw()


def press(app, button: int) -> None:
    assert app.handle_event(pygame.event.Event(pygame.CONTROLLERBUTTONDOWN, button=button))


def test_keyboard_drives_the_state_machine(app):
    run_for(app, 0.5)
    app.on_key(pygame.K_SPACE)
    run_for(app, 5)
    assert app.sim.state is State.ORBIT
    app.on_key(pygame.K_h)
    assert app.sim.state is State.HOLD
    app.on_key(pygame.K_h)
    assert app.sim.state is State.ORBIT
    app.on_key(pygame.K_l)
    assert app.sim.state is State.LAND
    assert app.on_key(pygame.K_ESCAPE) is False


def test_gamepad_buttons_drive_the_state_machine(app):
    run_for(app, 0.5)
    press(app, pygame.CONTROLLER_BUTTON_A)
    run_for(app, 5)
    assert app.sim.state is State.ORBIT
    press(app, pygame.CONTROLLER_BUTTON_Y)
    assert app.sim.state is State.HOLD
    press(app, pygame.CONTROLLER_BUTTON_Y)
    assert app.sim.state is State.ORBIT
    press(app, pygame.CONTROLLER_BUTTON_BACK)
    assert app.sim.tracker.dropout
    press(app, pygame.CONTROLLER_BUTTON_BACK)
    press(app, pygame.CONTROLLER_BUTTON_B)
    assert app.sim.state is State.LAND


@pytest.mark.parametrize("button", [pygame.CONTROLLER_BUTTON_LEFTSHOULDER, pygame.CONTROLLER_BUTTON_RIGHTSHOULDER])
def test_gamepad_shoulder_buttons_are_emergency_stop(app, button):
    run_for(app, 0.5)
    press(app, pygame.CONTROLLER_BUTTON_A)
    run_for(app, 5)
    press(app, button)
    assert app.sim.state is State.EMERGENCY_STOP
    assert not app.sim.drone.motors_on
    press(app, pygame.CONTROLLER_BUTTON_A)
    assert app.sim.state is State.EMERGENCY_STOP
    assert app.message.startswith("отклонено")
    press(app, pygame.CONTROLLER_BUTTON_START)
    assert app.sim.state is State.DISARMED


def test_losing_the_gamepad_in_flight_lands(app):
    run_for(app, 0.5)
    press(app, pygame.CONTROLLER_BUTTON_A)
    run_for(app, 5)
    app.on_gamepad_lost()
    assert app.sim.state is State.LAND
    assert "посадка" in app.message


def test_losing_the_gamepad_on_the_ground_changes_nothing(app):
    app.on_gamepad_lost()
    assert app.sim.state is State.DISARMED
