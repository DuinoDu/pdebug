"""Pointcloud process utils."""

from pdebug.data_types import PointcloudTensor
from pdebug.geometry import Plane3d, Vector3d, is_zero
from pdebug.utils.env import OPEN3D_INSTALLED

import numpy as np

if OPEN3D_INSTALLED:
    import open3d as o3d
    from open3d.geometry import OrientedBoundingBox


__all__ = ["get_bbox3d_points_from_4_points", "get_open3d_obb_from_4_points"]


def get_bbox3d_points_from_4_points(points, keep_perpendicular=False):
    """
    Get bbox3d eight points from four points.

    Args:
        points: four points, shape is [4, 3]
        keep_perpendicular: update points_2 and points_3 to make each plane perpendicular


    ///      o------ x     same as opencv
    ///     /|
    ///    / |
    ///   /  | z
    ///  y
    ///
    ///      0 ------------------- 1
    ///       /|                /|
    ///      / |               / |
    ///     /  |              /  |
    ///    /   |             /   |
    /// 2 ------------------- 7  |
    ///   |    |____________|____| 6
    ///   |   /3            |   /
    ///   |  /              |  /
    ///   | /               | /
    ///   |/                |/
    /// 5 ------------------- 4

    """
    assert points.shape == (4, 3)

    if keep_perpendicular:
        plane_xoy = Plane3d(points[0], points[1], points[2])
        points[3] = plane_xoy.get_nearest_point_on_normal(points[3])
        plane_xoz = Plane3d(points[0], points[1], points[3])
        points[2] = plane_xoz.get_nearest_point_on_normal(points[2])
        # two plane is perpendicular
        assert round(plane_xoy.normal.dot(plane_xoz.normal), 7) == 0

    points_new = np.zeros((8, 3), dtype=points.dtype)
    points_new[:4] = points
    points_new[7] = points[1] + (points[2] - points[0])
    points_new[5] = points[2] + (points[3] - points[0])
    points_new[6] = points[3] + (points[1] - points[0])
    points_new[4] = points_new[7] + (points[3] - points[0])
    return points_new


def get_open3d_obb_from_4_points(points):
    """
    Get open3d obb from 4 points.

    # TODO: Fix rotation bug
    """
    assert points.shape == (4, 3)
    points_8 = get_bbox3d_points_from_4_points(points, keep_perpendicular=True)

    extent_x = np.linalg.norm(points[1] - points[0], 2)
    extent_y = np.linalg.norm(points[2] - points[0], 2)
    extent_z = np.linalg.norm(points[3] - points[0], 2)
    extent = np.asarray([extent_x, extent_y, extent_z])

    center = points_8.mean(axis=0)

    vec_x = (points_8[1] - points_8[0]) / extent_x
    vec_y = (points_8[2] - points_8[0]) / extent_y
    vec_z = (points_8[3] - points_8[0]) / extent_z

    assert is_zero(Vector3d.from_numpy(vec_x).dot(Vector3d.from_numpy(vec_y)))
    assert is_zero(Vector3d.from_numpy(vec_x).dot(Vector3d.from_numpy(vec_z)))
    assert is_zero(Vector3d.from_numpy(vec_y).dot(Vector3d.from_numpy(vec_z)))

    R = np.asarray([vec_x, vec_y, vec_z])
    obb = OrientedBoundingBox(center, R, extent)

    # points_obb = np.asarray(obb.get_box_points())
    # print("points_8: \n", points_8)
    # print("points_obb: \n", points_obb)
    return obb
