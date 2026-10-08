import math

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from bubble_robot.tracking import apriltag_cam  # noqa: E402

WIDTH, HEIGHT = 1280, 720
TAG_SIZE_M = 0.16


def render(tmp_path, tag_id, distance_m, yaw_deg, camera_matrix):
    """Кадр с меткой в известной позе: проецируем углы и натягиваем на них картинку метки."""
    path = tmp_path / "tag.png"
    apriltag_cam.save_tag_image(tag_id, path)
    tag = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    q, n = apriltag_cam.TAG_QUIET_ZONE_PX, apriltag_cam.TAG_IMAGE_PX
    src = np.array([[q, q], [q + n, q], [q + n, q + n], [q, q + n]], dtype=np.float32)

    # Метка анфас: её ось x совпадает с осью x камеры, y и z развёрнуты; затем поворот вокруг вертикали.
    yaw = math.radians(yaw_deg)
    facing = np.diag([1.0, -1.0, -1.0])
    turn = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rvec = cv2.Rodrigues(turn @ facing)[0]
    tvec = np.array([0.1, -0.05, distance_m])
    dst = cv2.projectPoints(apriltag_cam.tag_corners_3d(TAG_SIZE_M), rvec, tvec, camera_matrix, None)[0]

    homography = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
    return cv2.warpPerspective(tag, homography, (WIDTH, HEIGHT), borderValue=255), tvec


@pytest.mark.parametrize("distance_m, yaw_deg", [(1.0, 0), (2.0, 30), (3.0, -45)])
def test_pose_is_recovered_from_synthetic_frame(tmp_path, distance_m, yaw_deg):
    camera_matrix = apriltag_cam.intrinsics_from_hfov(WIDTH, HEIGHT, 70.0)
    frame, tvec = render(tmp_path, 7, distance_m, yaw_deg, camera_matrix)

    poses = apriltag_cam.TagDetector(TAG_SIZE_M, camera_matrix).detect(frame)

    assert [p.tag_id for p in poses] == [7]
    pose = poses[0]
    assert pose.position == pytest.approx(tuple(tvec), rel=0.03, abs=0.01)
    assert pose.distance_m == pytest.approx(float(np.linalg.norm(tvec)), rel=0.03)
    # Метка смещена от оси камеры, поэтому наклон отличается от yaw на несколько градусов.
    assert pose.tilt_deg == pytest.approx(abs(yaw_deg), abs=8)


def test_empty_frame_has_no_tags():
    camera_matrix = apriltag_cam.intrinsics_from_hfov(WIDTH, HEIGHT, 70.0)
    frame = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    assert apriltag_cam.TagDetector(TAG_SIZE_M, camera_matrix).detect(frame) == []
