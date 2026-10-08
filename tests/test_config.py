import copy

import pytest
import yaml

from bubble_robot.config import DEFAULT_CONFIG_PATH, ConfigError, parse_config


@pytest.fixture
def raw():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_default_config_holds_hw80_envelope(cfg):
    s = cfg.safety
    assert s.min_human_distance_m == 1.5
    assert (s.altitude_min_m, s.altitude_max_m) == (0.5, 1.2)
    assert s.max_speed_mps == 0.3
    assert s.max_accel_mps2 == 0.3
    assert s.tracking_hold_after_s == 0.3
    assert s.tracking_land_after_s == 2.0


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("bubble", "radius_min_m", 1.0),  # кольцо ближе минимальной дистанции
        ("bubble", "target_radius_m", 2.5),  # цель вне кольца
        ("bubble", "flight_altitude_m", 1.5),  # выше safety-потолка
        ("safety", "tracking_hold_after_s", 3.0),  # Hold позже Land
        ("control", "land_speed_mps", 0.5),  # быстрее лимита скорости
        ("room", "width_m", 0.5),  # геозона пуста
    ],
)
def test_unsafe_config_is_rejected(raw, section, key, value):
    broken = copy.deepcopy(raw)
    broken[section][key] = value
    with pytest.raises(ConfigError):
        parse_config(broken)


def test_unknown_and_missing_keys_are_rejected(raw):
    extra = copy.deepcopy(raw)
    extra["safety"]["max_sped_mps"] = 9
    with pytest.raises(ConfigError):
        parse_config(extra)
    del raw["safety"]["max_speed_mps"]
    with pytest.raises(ConfigError):
        parse_config(raw)
