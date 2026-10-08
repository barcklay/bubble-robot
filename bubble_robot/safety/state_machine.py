"""Safety state machine (HW-84).

Единственное место, где решается, в каком режиме дрон. Разрешённые переходы заданы
таблицей ALLOWED; всё остальное блокируется и попадает в `rejected`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..config import SafetyConfig


class State(Enum):
    DISARMED = "DISARMED"
    TAKEOFF = "TAKEOFF"
    ORBIT = "ORBIT"
    HOLD = "HOLD"
    LAND = "LAND"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class Command(Enum):
    TAKEOFF = "TAKEOFF"
    HOLD = "HOLD"
    RESUME = "RESUME"
    LAND = "LAND"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RESET = "RESET"


S = State
ALLOWED: dict[State, frozenset[State]] = {
    S.DISARMED: frozenset({S.TAKEOFF, S.EMERGENCY_STOP}),
    S.TAKEOFF: frozenset({S.ORBIT, S.HOLD, S.LAND, S.EMERGENCY_STOP}),
    S.ORBIT: frozenset({S.HOLD, S.LAND, S.EMERGENCY_STOP}),
    S.HOLD: frozenset({S.ORBIT, S.LAND, S.EMERGENCY_STOP}),
    # Посадку нельзя отменить: только довести до конца или аварийно остановить.
    S.LAND: frozenset({S.DISARMED, S.EMERGENCY_STOP}),
    # Из аварийной остановки выходит только явный RESET.
    S.EMERGENCY_STOP: frozenset({S.DISARMED}),
}

AIRBORNE = frozenset({S.TAKEOFF, S.ORBIT, S.HOLD, S.LAND})

ALTITUDE_REACHED_TOLERANCE_M = 0.05
LANDED_ALTITUDE_M = 0.02


@dataclass(frozen=True)
class Inputs:
    tracking_age_s: float = math.inf  # время с последнего валидного сэмпла трекера
    altitude_m: float = 0.0
    battery: float = 1.0  # доля заряда 0..1
    human_distance_m: float = math.inf  # горизонтальная дистанция до человека


@dataclass(frozen=True)
class Transition:
    t: float
    source: State
    target: State
    reason: str
    accepted: bool = True


class SafetyStateMachine:
    def __init__(self, safety: SafetyConfig, flight_altitude_m: float):
        self.safety = safety
        self.flight_altitude_m = flight_altitude_m
        self.state = State.DISARMED
        self.manual_hold = False
        self.history: list[Transition] = []
        self.rejected: list[Transition] = []

    def _go(self, target: State, t: float, reason: str) -> bool:
        if target not in ALLOWED[self.state]:
            self._reject(target, t, f"переход запрещён ({reason})")
            return False
        self.history.append(Transition(t, self.state, target, reason))
        self.state = target
        if target in (S.DISARMED, S.LAND, S.EMERGENCY_STOP):
            self.manual_hold = False
        return True

    def _reject(self, target: State, t: float, reason: str) -> None:
        self.rejected.append(Transition(t, self.state, target, reason, accepted=False))

    def _tracking_fresh(self, inputs: Inputs) -> bool:
        return inputs.tracking_age_s <= self.safety.tracking_hold_after_s

    def command(self, cmd: Command, t: float, inputs: Inputs) -> bool:
        """Команда оператора. Возвращает False, если команда отклонена."""
        if cmd is Command.EMERGENCY_STOP:
            # Высший приоритет: принимается из любого состояния, без условий.
            return self.state is S.EMERGENCY_STOP or self._go(S.EMERGENCY_STOP, t, "команда EMERGENCY_STOP")

        if cmd is Command.RESET:
            if self.state is not S.EMERGENCY_STOP:
                self._reject(S.DISARMED, t, "RESET имеет смысл только после EMERGENCY_STOP")
                return False
            return self._go(S.DISARMED, t, "команда RESET")

        if cmd is Command.TAKEOFF:
            if self.state is not S.DISARMED:
                return self._go(S.TAKEOFF, t, "команда TAKEOFF")
            blocker = self._takeoff_blocker(inputs)
            if blocker:
                self._reject(S.TAKEOFF, t, blocker)
                return False
            return self._go(S.TAKEOFF, t, "команда TAKEOFF")

        if cmd is Command.HOLD:
            if self.state in (S.TAKEOFF, S.HOLD):
                # На взлёте сначала набираем высоту, потом зависаем.
                self.manual_hold = True
                return True
            if self._go(S.HOLD, t, "команда HOLD"):
                self.manual_hold = True
                return True
            return False

        if cmd is Command.RESUME:
            if self.state is S.TAKEOFF and self.manual_hold:
                self.manual_hold = False
                return True
            if self.state is not S.HOLD:
                self._reject(S.ORBIT, t, "RESUME имеет смысл только из HOLD")
                return False
            self.manual_hold = False
            if not self._tracking_fresh(inputs):
                self._reject(S.ORBIT, t, "трекинг не валиден, остаёмся в HOLD")
                return False
            return self._go(S.ORBIT, t, "команда RESUME")

        if cmd is Command.LAND:
            return self._go(S.LAND, t, "команда LAND")

        raise ValueError(f"неизвестная команда: {cmd}")

    def _takeoff_blocker(self, inputs: Inputs) -> str | None:
        if not self._tracking_fresh(inputs):
            return "взлёт запрещён: нет валидного трекинга"
        if inputs.battery <= self.safety.battery_land_below:
            return "взлёт запрещён: низкий заряд"
        if inputs.human_distance_m < self.safety.min_human_distance_m:
            return "взлёт запрещён: человек ближе минимальной дистанции"
        return None

    def update(self, t: float, inputs: Inputs) -> State:
        """Автоматические переходы. Вызывается на каждом такте управления."""
        s = self.safety
        state = self.state

        if state in (S.DISARMED, S.EMERGENCY_STOP):
            return state

        if state is S.LAND:
            if inputs.altitude_m <= LANDED_ALTITUDE_M:
                self._go(S.DISARMED, t, "посадка завершена")
            return self.state

        if inputs.battery <= s.battery_land_below:
            self._go(S.LAND, t, "низкий заряд")
        elif inputs.tracking_age_s > s.tracking_land_after_s:
            self._go(S.LAND, t, f"трекинг потерян дольше {s.tracking_land_after_s} с")
        elif state is S.TAKEOFF:
            if inputs.altitude_m >= self.flight_altitude_m - ALTITUDE_REACHED_TOLERANCE_M:
                if self.manual_hold:
                    self._go(S.HOLD, t, "высота набрана, ручной HOLD")
                elif not self._tracking_fresh(inputs):
                    self._go(S.HOLD, t, "высота набрана, трекинг не валиден")
                else:
                    self._go(S.ORBIT, t, "высота набрана")
        elif state is S.ORBIT:
            if not self._tracking_fresh(inputs):
                self._go(S.HOLD, t, f"трекинг потерян дольше {s.tracking_hold_after_s} с")
        elif state is S.HOLD:
            if not self.manual_hold and self._tracking_fresh(inputs):
                self._go(S.ORBIT, t, "трекинг восстановлен")
        return self.state
