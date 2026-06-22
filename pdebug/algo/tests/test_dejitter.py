import numpy as np

from pdebug.algo.dejitter import Evaluate, SimpleSmooth


def test_simple_smooth_reduces_box_scale_jitter():
    boxes = np.asarray(
        [
            [98, 51, 202, 149],
            [101, 49, 199, 151],
            [100, 50, 200, 150],
            [102, 52, 198, 148],
            [99, 48, 201, 152],
            [100, 51, 200, 149],
        ],
        dtype=np.float32,
    )

    smoothed = SimpleSmooth(period=3).boxes(boxes)

    assert smoothed.shape == boxes.shape
    np.testing.assert_allclose(smoothed[0], boxes[0])
    np.testing.assert_allclose(smoothed[2], boxes[:3].mean(axis=0), rtol=1e-6)

    raw_error = Evaluate.boxes_scale_ratio(boxes)
    smoothed_error = Evaluate.boxes_scale_ratio(smoothed)

    assert smoothed_error < raw_error


def test_keypoint_smoothing_reduces_position_jitter_for_static_subject():
    keypoints = np.asarray(
        [
            [120, 70, 2, 180, 70, 2, 150, 120, 2],
            [122, 69, 2, 178, 72, 2, 151, 119, 2],
            [119, 71, 2, 181, 68, 2, 149, 121, 2],
            [121, 72, 2, 179, 69, 2, 152, 118, 2],
            [118, 68, 2, 182, 71, 2, 148, 122, 2],
            [120, 70, 2, 180, 70, 2, 150, 120, 2],
        ],
        dtype=np.float32,
    )
    boxes = np.asarray(
        [[95, 45, 205, 155]] * len(keypoints),
        dtype=np.float32,
    )

    smoothed = SimpleSmooth(period=3).keypoints(keypoints)

    assert smoothed.shape == keypoints.shape
    np.testing.assert_allclose(smoothed[:, 2::3], keypoints[:, 2::3])
    np.testing.assert_allclose(
        smoothed[3], keypoints[1:4].mean(axis=0), rtol=1e-6
    )

    raw_error = Evaluate.keypoints(keypoints, gt_boxes=boxes)
    smoothed_error = Evaluate.keypoints(smoothed, gt_boxes=boxes)

    assert smoothed_error < raw_error


def test_evaluate_box_center_returns_mean_box_ground_truth():
    boxes = np.asarray(
        [
            [9, 9, 21, 21],
            [11, 10, 23, 22],
            [10, 11, 22, 23],
        ],
        dtype=np.float32,
    )

    error, gt_boxes = Evaluate.boxes_center(boxes)

    np.testing.assert_allclose(
        gt_boxes, np.repeat(boxes.mean(axis=0)[None, :], len(boxes), axis=0)
    )
    assert error >= 0
