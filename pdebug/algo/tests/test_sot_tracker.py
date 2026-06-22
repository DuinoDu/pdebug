import numpy as np

from pdebug.algo.sot.tracker import BBoxTracker, PointTracker


def test_point_tracker_anchor_converts_points_to_tracker_boxes():
    tracker = PointTracker("MIL", point_size=4)
    points = np.asarray([[20.8, 30.2], [100.1, 50.9]], dtype=np.float32)

    tracker.set_anchor(points)

    np.testing.assert_array_equal(
        tracker.boxes,
        np.asarray([[16, 26, 8, 8], [96, 46, 8, 8]]),
    )
    assert np.issubdtype(tracker.boxes.dtype, np.integer)


def test_point_tracker_anchor_accepts_single_point():
    tracker = PointTracker("MIL", point_size=5)

    tracker.set_anchor(np.asarray([40.0, 25.0], dtype=np.float32))

    np.testing.assert_array_equal(
        tracker.boxes, np.asarray([[35, 20, 10, 10]])
    )


def test_bbox_tracker_anchor_normalizes_single_and_multiple_boxes():
    single_tracker = BBoxTracker("MIL")
    single_tracker.set_anchor(np.asarray([10.7, 20.2, 30.9, 40.1]))

    np.testing.assert_array_equal(
        single_tracker.boxes, np.asarray([[10, 20, 30, 40]])
    )

    multi_tracker = BBoxTracker("MIL")
    multi_tracker.set_anchor(
        np.asarray(
            [
                [0, 0, 15, 20],
                [50.5, 60.5, 30.2, 35.8],
            ]
        )
    )

    np.testing.assert_array_equal(
        multi_tracker.boxes,
        np.asarray([[0, 0, 15, 20], [50, 60, 30, 35]]),
    )
    assert np.issubdtype(multi_tracker.boxes.dtype, np.integer)
