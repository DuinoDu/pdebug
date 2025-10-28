from typing import Optional

import numpy as np

__all__ = [
    "points_to_keypoints",
    "keypoints_to_points",
    "points_to_boxes",
    "boxes_to_points",
]


BAD_POINT_VALUE = -1


def points_to_keypoints(
    points: np.ndarray, vis: Optional[int] = 2
) -> np.ndarray:
    """Convert points to keypoints."""
    assert points.ndim == 2
    assert points.shape[1] == 2
    x = points[:, 0]
    y = points[:, 1]
    v = np.zeros_like(x)
    v.fill(vis)
    bad_index = np.where(x == BAD_POINT_VALUE)[0]
    v[bad_index] = 0.0
    keypoints = np.concatenate((x[:, None], y[:, None], v[:, None]), axis=1)
    return keypoints.flatten()[None, :]


def keypoints_to_points(keypoints: np.ndarray) -> np.ndarray:
    """Convert keypoints to points.

    Args:
        keypoints: [N, K*3], 3 means (x, y, vis)

    Returns:
        points: [N, K*2]

    """
    assert keypoints.ndim == 2
    N = keypoints.shape[0]
    x = keypoints[:, 0::3]
    y = keypoints[:, 1::3]
    points = np.concatenate((x[:, :, None], y[:, :, None]), axis=2)
    points = points.reshape(N, -1)
    return points


def points_to_boxes(
    points: np.ndarray, point_size: int, layout="x1y1wh"
) -> np.ndarray:
    """Convert points to boxes."""
    assert layout == "x1y1wh"
    X1 = points[:, 0] - point_size
    Y1 = points[:, 1] - point_size
    X2 = points[:, 0] + point_size
    Y2 = points[:, 1] + point_size
    W = X2 - X1
    H = Y2 - Y1
    boxes = np.concatenate(
        (X1[:, None], Y1[:, None], W[:, None], H[:, None]), axis=1
    )
    return boxes


def boxes_to_points(boxes: np.ndarray, point_size: int):
    """Convert boxes to points."""
    boxes = np.asarray(boxes)
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4
    points_X = boxes[:, 0] + point_size
    points_Y = boxes[:, 1] + point_size
    bad_idx = np.logical_and(boxes[:, 0] == 0, boxes[:, 1] == 0)
    points_X[bad_idx] = BAD_POINT_VALUE
    points_Y[bad_idx] = BAD_POINT_VALUE
    points = np.concatenate((points_X[:, None], points_Y[:, None]), axis=1)
    return points
