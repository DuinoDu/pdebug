from pdebug.piata import Input, Output
from pdebug.piata.type_cast import (
    boxes_to_points,
    keypoints_to_points,
    points_to_boxes,
    points_to_keypoints,
)

import cv2
import numpy as np


def test_imgdir_output_can_be_read_back_as_bgr_frames(tmp_path):
    first_frame = np.zeros((6, 8, 3), dtype=np.uint8)
    first_frame[1:5, 2:6] = [20, 120, 240]
    second_frame = np.full((6, 8, 3), [80, 40, 10], dtype=np.uint8)
    frames = [first_frame, second_frame]
    output_dir = tmp_path / "frames"

    Output(frames, name="imgdir", ext=".png").save(str(output_dir))
    reader = Input(output_dir, name="imgdir").get_reader()

    assert len(reader) == 2
    loaded = [next(reader) for _ in range(len(reader))]
    np.testing.assert_array_equal(loaded[0], first_frame)
    np.testing.assert_array_equal(loaded[1], second_frame)

    reader.reset()
    assert reader.idx == 0
    assert reader.filename is None


def test_imgdir_reader_converts_camera_frames_to_rgb_when_requested(tmp_path):
    bgr_frame = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr_frame[:, :] = [15, 90, 210]
    image_path = tmp_path / "camera.png"
    cv2.imwrite(str(image_path), bgr_frame)

    reader = Input(tmp_path, name="imgdir", to_rgb=True).get_reader()

    rgb_frame = next(reader)
    expected_rgb = np.full_like(bgr_frame, [210, 90, 15])
    np.testing.assert_array_equal(rgb_frame, expected_rgb)


def test_points_and_keypoints_roundtrip_for_landmark_annotations():
    points = np.asarray(
        [
            [320.0, 240.0],
            [360.0, 245.0],
            [340.0, 290.0],
        ],
        dtype=np.float32,
    )

    keypoints = points_to_keypoints(points, vis=2)
    restored_points = keypoints_to_points(keypoints)

    assert keypoints.shape == (1, 9)
    np.testing.assert_array_equal(keypoints[0, 2::3], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(restored_points, points.reshape(1, -1))


def test_points_and_tracker_boxes_roundtrip_for_click_tracking():
    points = np.asarray(
        [
            [50.0, 75.0],
            [125.0, 160.0],
        ],
        dtype=np.float32,
    )

    boxes = points_to_boxes(points, point_size=6, layout="x1y1wh")
    restored_points = boxes_to_points(boxes, point_size=6)

    np.testing.assert_allclose(
        boxes,
        np.asarray(
            [
                [44.0, 69.0, 12.0, 12.0],
                [119.0, 154.0, 12.0, 12.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(restored_points, points)
