"""AprilTag 36h11 на обычной камере (HW-85): детекция, поза и замер условий потери.

Без калибровки камеры матрица строится из угла обзора (--hfov), поэтому абсолютная
дистанция верна с точностью до ошибки этого угла. Частота, углы и сам факт потери
от этого не зависят.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
TAG_IMAGE_PX = 800
TAG_QUIET_ZONE_PX = 100


@dataclass(frozen=True)
class TagPose:
    tag_id: int
    position: tuple[float, float, float]  # метка в системе камеры: x вправо, y вниз, z от камеры, м
    rvec: tuple[float, float, float]  # ориентация метки, вектор Родригеса
    distance_m: float
    tilt_deg: float  # угол между нормалью метки и направлением на камеру; 0 — метка анфас


def save_tag_image(tag_id: int, path: str | Path) -> None:
    """PNG для печати. Размер метки (--tag-size) — сторона чёрного квадрата, без белых полей."""
    tag = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, TAG_IMAGE_PX)
    cv2.imwrite(str(path), cv2.copyMakeBorder(tag, *[TAG_QUIET_ZONE_PX] * 4, cv2.BORDER_CONSTANT, value=255))


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def tag_corners_3d(tag_size_m: float) -> np.ndarray:
    h = tag_size_m / 2
    # Порядок углов как у детектора: верхний левый, далее по часовой стрелке.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


class TagDetector:
    def __init__(self, tag_size_m: float, camera_matrix: np.ndarray):
        self.camera_matrix = camera_matrix
        self.object_points = tag_corners_3d(tag_size_m)
        self.detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())

    def detect(self, gray: np.ndarray) -> list[TagPose]:
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        poses = []
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points,
                tag_corners.reshape(4, 2).astype(np.float64),
                self.camera_matrix,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            t = tvec.flatten()
            distance = float(np.linalg.norm(t))
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            cos_tilt = abs(float(normal @ t)) / distance
            poses.append(
                TagPose(
                    tag_id=int(tag_id),
                    position=(float(t[0]), float(t[1]), float(t[2])),
                    rvec=tuple(float(v) for v in rvec.flatten()),
                    distance_m=distance,
                    tilt_deg=math.degrees(math.acos(min(1.0, cos_tilt))),
                )
            )
        return poses


def run_live(camera_index: int, tag_size_m: float, hfov_deg: float, log_dir: str) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"камера {camera_index} не открылась — проверь доступ терминала к камере в настройках macOS")
        return 1
    ok, frame = cap.read()
    if not ok:
        print("камера открылась, но кадр не пришёл")
        return 1
    height, width = frame.shape[:2]
    detector = TagDetector(tag_size_m, intrinsics_from_hfov(width, height, hfov_deg))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / datetime.now().strftime("apriltag-%Y%m%d-%H%M%S.csv")
    frames = detected_frames = 0
    losses: list[TagPose] = []  # последняя поза перед каждой потерей метки
    max_distance = max_tilt = 0.0
    last_pose: TagPose | None = None
    started = time.monotonic()
    fps = 0.0
    print(f"камера {width}×{height}, метка {tag_size_m} м. Q или Esc — закончить и показать итог.")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "detected", "id", "x", "y", "z", "distance", "tilt_deg", "rx", "ry", "rz", "fps"])
        while True:
            tick = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            poses = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            pose = poses[0] if poses else None
            frames += 1
            t = tick - started
            if pose:
                detected_frames += 1
                max_distance = max(max_distance, pose.distance_m)
                max_tilt = max(max_tilt, pose.tilt_deg)
                writer.writerow(
                    [f"{t:.3f}", 1, pose.tag_id, *(f"{v:.4f}" for v in pose.position)]
                    + [f"{pose.distance_m:.4f}", f"{pose.tilt_deg:.1f}", *(f"{v:.4f}" for v in pose.rvec), f"{fps:.1f}"]
                )
                label = f"id {pose.tag_id}  {pose.distance_m:.2f} m  tilt {pose.tilt_deg:.0f} deg"
            else:
                if last_pose:
                    losses.append(last_pose)
                writer.writerow([f"{t:.3f}", 0, "", "", "", "", "", "", "", "", "", f"{fps:.1f}"])
                label = "no tag"
            last_pose = pose

            cv2.putText(frame, f"{fps:4.1f} FPS  {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("AprilTag", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame_time = time.monotonic() - tick
            fps = 0.9 * fps + 0.1 / frame_time if fps else 1 / frame_time

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.monotonic() - started
    print(f"кадров: {frames} за {elapsed:.1f} с — в среднем {frames / elapsed:.1f} FPS (цель ≥20)")
    print(f"метка видна в {100 * detected_frames / max(frames, 1):.0f} % кадров")
    print(f"макс. дистанция с детекцией: {max_distance:.2f} м, макс. наклон: {max_tilt:.0f}°")
    print(f"потерь метки: {len(losses)}")
    for pose in losses[:20]:
        print(f"  потеря на {pose.distance_m:.2f} м, наклон {pose.tilt_deg:.0f}°")
    print(f"лог: {log_path}")
    return 0
