"""Окно симулятора: вид комнаты сверху и панель состояния."""

from __future__ import annotations

import math

import pygame

from ..safety.state_machine import Command, State
from ..sim.scenario import DemoScenario
from ..sim.world import Simulation

PX_PER_M = 100
PAD = 20
PANEL_W = 300
TRAIL_LEN = 1500

BG = (18, 20, 26)
ROOM = (32, 36, 46)
GRID = (44, 49, 62)
GEOFENCE = (90, 98, 120)
TEXT = (225, 228, 235)
MUTED = (140, 147, 165)
HUMAN = (240, 240, 245)
FORBIDDEN = (200, 70, 70)
BAND = (70, 130, 110)
TARGET = (250, 210, 90)
TRAIL = (80, 110, 160)
BAD = (235, 90, 90)

STATE_COLORS = {
    State.DISARMED: (130, 136, 150),
    State.TAKEOFF: (100, 170, 250),
    State.ORBIT: (90, 210, 140),
    State.HOLD: (250, 200, 80),
    State.LAND: (240, 150, 70),
    State.EMERGENCY_STOP: (235, 70, 70),
}

HELP = [
    ("WASD / стрелки", "человек"),
    ("Space", "взлёт"),
    ("H", "Hold / Resume"),
    ("L", "посадка"),
    ("E", "EMERGENCY STOP"),
    ("R", "сброс после аварии"),
    ("T", "потеря трекинга"),
    ("Esc", "выход"),
]


class SimApp:
    def __init__(self, sim: Simulation, scenario: DemoScenario | None = None):
        self.sim = sim
        self.scenario = scenario
        room = sim.cfg.room
        self.room_px = (int(room.width_m * PX_PER_M), int(room.depth_m * PX_PER_M))
        self.size = (self.room_px[0] + PANEL_W + PAD * 3, self.room_px[1] + PAD * 2)
        self.trail: list[tuple[int, int]] = []
        self.message = ""
        pygame.init()
        pygame.display.set_caption("Bubble Robot — симулятор")
        self.screen = pygame.display.set_mode(self.size)
        font_name = pygame.font.match_font("menlo,dejavusansmono,arial")
        self.font = pygame.font.Font(font_name, 15)
        self.font_big = pygame.font.Font(font_name, 26)

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(PAD + self.room_px[0] / 2 + x * PX_PER_M),
            int(PAD + self.room_px[1] / 2 - y * PX_PER_M),
        )

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.on_key(event.key)
            if self.scenario is not None:
                self.scenario.apply(self.sim)
                running = running and self.sim.t < self.scenario.duration_s
            else:
                self.sim.set_human_velocity(*self.human_input())
            self.sim.step()
            self.draw()
            pygame.display.flip()
            clock.tick(round(1 / self.sim.dt))
        pygame.quit()

    def human_input(self) -> tuple[float, float]:
        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        vy = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        speed = self.sim.cfg.sim.human_speed_mps
        return vx * speed, vy * speed

    def on_key(self, key: int) -> bool:
        sim = self.sim
        if key == pygame.K_ESCAPE:
            return False
        command = None
        if key == pygame.K_SPACE:
            command = Command.TAKEOFF
        elif key == pygame.K_h:
            command = Command.RESUME if sim.state is State.HOLD and sim.sm.manual_hold else Command.HOLD
        elif key == pygame.K_l:
            command = Command.LAND
        elif key == pygame.K_e:
            command = Command.EMERGENCY_STOP
        elif key == pygame.K_r:
            command = Command.RESET
        elif key == pygame.K_t:
            sim.tracker.dropout = not sim.tracker.dropout
        if command is not None:
            accepted = sim.command(command)
            self.message = "" if accepted else f"отклонено: {sim.sm.rejected[-1].reason}"
        return True

    # --- отрисовка ----------------------------------------------------------

    def draw(self) -> None:
        sim, screen = self.sim, self.screen
        screen.fill(BG)
        room_rect = pygame.Rect(PAD, PAD, *self.room_px)
        pygame.draw.rect(screen, ROOM, room_rect)
        for i in range(1, int(sim.cfg.room.width_m)):
            x = PAD + i * PX_PER_M
            pygame.draw.line(screen, GRID, (x, PAD), (x, PAD + self.room_px[1]))
        for i in range(1, int(sim.cfg.room.depth_m)):
            y = PAD + i * PX_PER_M
            pygame.draw.line(screen, GRID, (PAD, y), (PAD + self.room_px[0], y))
        margin = int(sim.cfg.safety.geofence_margin_m * PX_PER_M)
        pygame.draw.rect(screen, GEOFENCE, room_rect.inflate(-2 * margin, -2 * margin), 1)

        screen.set_clip(room_rect)
        human_px = self.to_px(*sim.human)
        bubble = sim.cfg.bubble
        band_w = int((bubble.radius_max_m - bubble.radius_min_m) * PX_PER_M)
        pygame.draw.circle(screen, BAND, human_px, int(bubble.radius_max_m * PX_PER_M), band_w)
        pygame.draw.circle(screen, FORBIDDEN, human_px, int(sim.cfg.safety.min_human_distance_m * PX_PER_M), 2)

        drone_px = self.to_px(sim.drone.pos[0], sim.drone.pos[1])
        self.trail.append(drone_px)
        del self.trail[:-TRAIL_LEN]
        if len(self.trail) > 1:
            pygame.draw.lines(screen, TRAIL, False, self.trail, 1)

        if sim.target is not None:
            tx, ty = self.to_px(*sim.target)
            pygame.draw.line(screen, TARGET, (tx - 6, ty), (tx + 6, ty), 2)
            pygame.draw.line(screen, TARGET, (tx, ty - 6), (tx, ty + 6), 2)

        pygame.draw.circle(screen, HUMAN, human_px, 12)
        nose = (
            human_px[0] + int(22 * math.cos(sim.human_heading)),
            human_px[1] - int(22 * math.sin(sim.human_heading)),
        )
        pygame.draw.line(screen, HUMAN, human_px, nose, 3)

        color = STATE_COLORS[sim.state]
        pygame.draw.circle(screen, color, drone_px, 9)
        pygame.draw.circle(screen, color, drone_px, 9 + int(sim.drone.pos[2] * 10), 1)
        screen.set_clip(None)

        self.draw_panel()

    def draw_panel(self) -> None:
        sim, screen = self.sim, self.screen
        x = self.room_px[0] + PAD * 2
        y = PAD
        screen.blit(self.font_big.render(sim.state.value, True, STATE_COLORS[sim.state]), (x, y))
        y += 44

        safety = sim.cfg.safety
        speed = math.sqrt(sum(v * v for v in sim.drone.vel))
        age = sim.tracking_age
        tracking = "нет" if not math.isfinite(age) else f"{age:.2f} с"
        rows = [
            ("время", f"{sim.t:6.1f} с", False),
            ("дистанция", f"{sim.distance:.2f} м", sim.distance < safety.min_human_distance_m),
            ("скорость", f"{speed:.2f} м/с", speed > safety.max_speed_mps * 1.02),
            ("высота", f"{sim.drone.pos[2]:.2f} м", sim.drone.pos[2] > safety.altitude_max_m),
            ("трекинг", tracking, age > safety.tracking_hold_after_s),
            ("батарея", f"{sim.drone.battery * 100:.0f} %", sim.drone.battery <= safety.battery_land_below),
            ("нарушений", str(sim.stats.violations), sim.stats.violations > 0),
        ]
        for label, value, bad in rows:
            screen.blit(self.font.render(label, True, MUTED), (x, y))
            screen.blit(self.font.render(value, True, BAD if bad else TEXT), (x + 120, y))
            y += 24
        if sim.tracker.dropout:
            screen.blit(self.font.render("трекер выключен (T)", True, BAD), (x, y))
        y += 34

        for key, action in HELP:
            screen.blit(self.font.render(key, True, TEXT), (x, y))
            screen.blit(self.font.render(action, True, MUTED), (x + 140, y))
            y += 22
        if self.message:
            y += 12
            for line in _wrap(self.message, 32):
                screen.blit(self.font.render(line, True, BAD), (x, y))
                y += 20


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]
