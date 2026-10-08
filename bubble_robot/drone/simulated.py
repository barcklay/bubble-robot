"""Симулированный дрон: точечная масса с отложенным откликом на velocity setpoint."""

from __future__ import annotations

from ..config import SimConfig

GRAVITY = 9.81


class SimulatedDrone:
    def __init__(self, sim: SimConfig):
        self.tau = sim.drone_response_s
        self.drain_per_s = 1.0 / sim.battery_flight_time_s
        self.pos = [sim.drone_start[0], sim.drone_start[1], 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.battery = 1.0
        self.motors_on = False
        self._setpoint = (0.0, 0.0, 0.0)

    def start_motors(self) -> None:
        self.motors_on = True

    def stop_motors(self) -> None:
        self.motors_on = False
        self._setpoint = (0.0, 0.0, 0.0)

    def send_velocity_setpoint(self, v: tuple[float, float, float]) -> None:
        self._setpoint = v

    def step(self, dt: float) -> None:
        if self.motors_on:
            k = min(1.0, dt / self.tau)
            for i in range(3):
                self.vel[i] += (self._setpoint[i] - self.vel[i]) * k
            self.battery = max(0.0, self.battery - self.drain_per_s * dt)
        else:
            # Без тяги дрон падает вертикально.
            self.vel[0] = self.vel[1] = 0.0
            self.vel[2] -= GRAVITY * dt
        for i in range(3):
            self.pos[i] += self.vel[i] * dt
        if self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.vel[2] = max(0.0, self.vel[2])
            if not self.motors_on:
                self.vel[2] = 0.0
