"""Точка входа: `./run` открывает симулятор, `./run --help` показывает остальное."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .logs import RunLogger
from .safety.state_machine import State
from .sim.scenario import DemoScenario, run_scenario
from .sim.world import Simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble_robot", description="Bubble Robot — Flying Companion MVP")
    parser.add_argument("--config", help="путь к config.yaml (по умолчанию — из корня проекта)")
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("sim", help="симулятор bubble-поведения (по умолчанию)")
    sim.add_argument("--scenario", action="store_true", help="скриптовый демо-сценарий вместо клавиатуры")
    sim.add_argument("--headless", action="store_true", help="сценарий без окна, быстрее реального времени")
    sim.add_argument("--duration", type=float, default=300.0, help="длительность сценария, с")
    sim.add_argument("--seed", type=int, default=0)

    tag = sub.add_parser("apriltag", help="проверка AprilTag-трекинга на камере (HW-85)")
    tag.add_argument("--tag-size", type=float, required=True, help="сторона чёрного квадрата метки, м")
    tag.add_argument("--camera", type=int, default=0, help="индекс камеры")
    tag.add_argument("--hfov", type=float, default=70.0, help="горизонтальный угол обзора камеры, градусы")

    make = sub.add_parser("make-tag", help="PNG с меткой AprilTag 36h11 для печати")
    make.add_argument("--id", type=int, default=0)
    make.add_argument("--out", default="apriltag.png")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "apriltag":
        from .tracking import apriltag_cam

        return apriltag_cam.run_live(args.camera, args.tag_size, args.hfov, cfg.logging.dir)
    if args.command == "make-tag":
        from .tracking import apriltag_cam

        apriltag_cam.save_tag_image(args.id, args.out)
        print(f"метка id={args.id} сохранена: {args.out}")
        return 0
    return run_sim(cfg, args)


def run_sim(cfg, args) -> int:
    headless = getattr(args, "headless", False)
    scripted = headless or getattr(args, "scenario", False)
    logger = RunLogger(cfg.logging.dir)
    sim = Simulation(cfg, logger, seed=getattr(args, "seed", 0))
    scenario = DemoScenario(getattr(args, "duration", 300.0)) if scripted else None
    try:
        if headless:
            run_scenario(sim, scenario)
        else:
            from .ui.sim_app import SimApp

            SimApp(sim, scenario).run()
    finally:
        logger.close()
    print_summary(sim, logger)
    return 1 if sim.stats.violations else 0


def print_summary(sim: Simulation, logger: RunLogger) -> None:
    stats = sim.stats
    orbit_ticks = stats.state_ticks[State.ORBIT]
    print(f"длительность:        {sim.t:.1f} с")
    for state in State:
        if stats.state_ticks[state]:
            print(f"  {state.value:<15} {stats.state_ticks[state] * sim.dt:6.1f} с")
    if stats.min_distance_m < float("inf"):
        print(f"мин. дистанция:      {stats.min_distance_m:.2f} м (предел {sim.cfg.safety.min_human_distance_m})")
        print(f"макс. скорость:      {stats.max_speed_mps:.2f} м/с (предел {sim.cfg.safety.max_speed_mps})")
    if orbit_ticks:
        print(f"в кольце 1.5–2 м:    {100 * stats.orbit_in_band_ticks / orbit_ticks:.1f} % времени ORBIT")
    if stats.violations:
        detail = ", ".join(f"{code}: {n * sim.dt:.1f} с" for code, n in stats.violation_ticks.items())
        print(f"НАРУШЕНИЯ:           {detail}")
    else:
        print("нарушений:           нет")
    print(f"лог:                 {logger.telemetry_path}")
    print(f"события:             {logger.events_path}")


if __name__ == "__main__":
    sys.exit(main())
