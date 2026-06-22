from pathlib import Path

import numpy as np

from pdebug.piata import Input, Output
from pdebug.piata.type_cast import (
    BAD_POINT_VALUE,
    boxes_to_points,
    keypoints_to_points,
    points_to_boxes,
    points_to_keypoints,
)


def _sample_bgr_images():
    first = np.full((5, 7, 3), (10, 20, 30), dtype=np.uint8)
    second = np.zeros((5, 7, 3), dtype=np.uint8)
    second[:, :, 0] = 80
    second[:, :, 1] = np.arange(7, dtype=np.uint8)
    second[:, :, 2] = np.arange(5, dtype=np.uint8)[:, None]
    return [first, second]


def test_imgdir_output_can_be_read_back_with_input(tmp_path):
    images = _sample_bgr_images()
    imgdir = tmp_path / "frames"

    Output(images, name="imgdir", ext=".png").save(str(imgdir))

    assert sorted(path.name for path in imgdir.iterdir()) == [
        "0.png",
        "1.png",
    ]

    reader = Input(str(imgdir), name="imgdir").get_reader()

    assert len(reader) == 2
    assert [Path(path).name for path in reader.imgfiles] == ["0.png", "1.png"]

    read_back = [next(reader) for _ in range(len(reader))]
    for actual, expected in zip(read_back, images):
        np.testing.assert_array_equal(actual, expected)

    reader.reset()
    np.testing.assert_array_equal(next(reader), images[0])
    np.testing.assert_array_equal(reader.imread("1.png"), images[1])


def test_type_cast_converts_points_keypoints_and_boxes_for_annotations():
    landmark_points = np.array(
        [
            [12.0, 18.0],
            [44.5, 32.0],
            [BAD_POINT_VALUE, BAD_POINT_VALUE],
        ],
        dtype=np.float32,
    )

    keypoints = points_to_keypoints(landmark_points, vis=2)

    np.testing.assert_allclose(
        keypoints,
        np.array(
            [[12.0, 18.0, 2.0, 44.5, 32.0, 2.0, -1.0, -1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        keypoints_to_points(keypoints),
        np.array([[12.0, 18.0, 44.5, 32.0, -1.0, -1.0]], dtype=np.float32),
    )

    object_centers = np.array(
        [[32.0, 40.0], [96.0, 72.0]],
        dtype=np.float32,
    )
    boxes = points_to_boxes(object_centers, point_size=6)

    np.testing.assert_allclose(
        boxes,
        np.array([[26.0, 34.0, 12.0, 12.0], [90.0, 66.0, 12.0, 12.0]]),
    )
    np.testing.assert_allclose(
        boxes_to_points(boxes, point_size=6),
        object_centers,
    )
