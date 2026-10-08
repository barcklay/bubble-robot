"""Загрузка и проверка config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SafetyConfig:
    min_human_distance_m: float
    altitude_min_m: float
    altitude_max_m: float
    max_speed_mps: float
    max_accel_mps2: float
    tracking_hold_after_s: float
    tracking_land_after_s: float
    battery_land_below: float
    geofence_margin_m: float


@dataclass(frozen=True)
class BubbleConfig:
    radius_min_m: float
    radius_max_m: float
    target_radius_m: float
    side_deg: float
    flight_altitude_m: float


@dataclass(frozen=True)
class RoomConfig:
    width_m: float
    depth_m: float


@dataclass(frozen=True)
class ControlConfig:
    rate_hz: float
    radial_gain: float
    tangential_gain: float
    altitude_gain: float
    land_speed_mps: float


@dataclass(frozen=True)
class SimConfig:
    human_speed_mps: float
    human_accel_mps2: float
    drone_start: tuple[float, float]
    drone_response_s: float
    battery_flight_time_s: float
    tracker_rate_hz: float
    tracker_latency_s: float
    tracker_noise_m: float


@dataclass(frozen=True)
class LoggingConfig:
    dir: str


@dataclass(frozen=True)
class Config:
    safety: SafetyConfig
    bubble: BubbleConfig
    room: RoomConfig
    control: ControlConfig
    sim: SimConfig
    logging: LoggingConfig

    @property
    def geofence_half_extents(self) -> tuple[float, float]:
        m = self.safety.geofence_margin_m
        return self.room.width_m / 2 - m, self.room.depth_m / 2 - m


def _section(cls, data: dict, name: str):
    raw = data.get(name)
    if not isinstance(raw, dict):
        raise ConfigError(f"в конфиге нет раздела '{name}'")
    expected = {f.name for f in fields(cls)}
    if set(raw) != expected:
        missing, extra = expected - set(raw), set(raw) - expected
        raise ConfigError(f"раздел '{name}': не хватает {sorted(missing)}, лишние {sorted(extra)}")
    if cls is SimConfig:
        raw = {**raw, "drone_start": tuple(raw["drone_start"])}
    return cls(**raw)


def parse_config(data: dict) -> Config:
    cfg = Config(
        safety=_section(SafetyConfig, data, "safety"),
        bubble=_section(BubbleConfig, data, "bubble"),
        room=_section(RoomConfig, data, "room"),
        control=_section(ControlConfig, data, "control"),
        sim=_section(SimConfig, data, "sim"),
        logging=_section(LoggingConfig, data, "logging"),
    )
    _validate(cfg)
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f))


def _validate(cfg: Config) -> None:
    s, b = cfg.safety, cfg.bubble
    checks = [
        (s.max_speed_mps > 0 and s.max_accel_mps2 > 0, "скорость и ускорение должны быть > 0"),
        (0 < s.altitude_min_m < s.altitude_max_m, "altitude_min_m должна быть меньше altitude_max_m"),
        (
            0 < s.tracking_hold_after_s < s.tracking_land_after_s,
            "tracking_hold_after_s должен быть меньше tracking_land_after_s",
        ),
        (0 <= s.battery_land_below < 1, "battery_land_below должен быть в [0, 1)"),
        (
            b.radius_min_m >= s.min_human_distance_m,
            "bubble.radius_min_m не может быть меньше safety.min_human_distance_m",
        ),
        (
            b.radius_min_m <= b.target_radius_m <= b.radius_max_m,
            "target_radius_m должен лежать в [radius_min_m, radius_max_m]",
        ),
        (
            s.altitude_min_m <= b.flight_altitude_m <= s.altitude_max_m,
            "flight_altitude_m должна лежать в safety-диапазоне высот",
        ),
        (cfg.control.land_speed_mps <= s.max_speed_mps, "land_speed_mps превышает max_speed_mps"),
        (min(cfg.geofence_half_extents) > 0, "геозона пуста: комната меньше двух отступов"),
        (cfg.control.rate_hz > 0 and cfg.sim.tracker_rate_hz > 0, "частоты должны быть > 0"),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)
