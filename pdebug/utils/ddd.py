"""3d related utilities."""
from typing import Dict, List, Optional, Tuple

from pdebug.data_types import Camera, PointcloudTensor, Tensor
from pdebug.utils.env import OPEN3D_INSTALLED, TRIMESH_INSTALLED

import cv2
import numpy as np

if OPEN3D_INSTALLED:
    import open3d as o3d
else:
    o3d = None

if TRIMESH_INSTALLED:
    import trimesh


__all__ = [
    "align_size",
    "generate_pcd_from_depth",
    "merge_pointcloud",
    "align_timestamp",
    "project_depth_to_rgb",
    "transform_points",
    "get_3D_corners",
    "load_points_3d_from_cad",
]


def align_size(x: Tensor, y: Tensor, factor=1.0) -> Tuple[Tensor, float]:
    """Alias x shape to y."""
    src_h, src_w = x.shape[:2]
    if y is None:
        dst_h, dst_w = int(src_h * factor), int(src_w * factor)
    else:
        dst_h, dst_w = y.shape[:2]
    if src_h != dst_h or src_w != dst_w:
        x = cv2.resize(x, (dst_w, dst_h))
        factor = dst_w / src_w
    return x, factor


def generate_pcd_from_depth_open3d(
    depth: Tensor,
    camera: Camera,
    rgb: Optional[Tensor] = None,
    depth_rgb_scale: Optional[float] = None,
) -> Optional[PointcloudTensor]:
    """Generate pcd from depthmap.

    Args:
        depth: input depth data.
        camera: camera param. If rgb provided, this camera should be rgb camera.
        rgb: input rgb data, used as pointcloud color.
        depth_rgb_scale: depth and rgb scale factor, only used when rgb is None.
    """
    assert OPEN3D_INSTALLED, "open3d is required."

    if rgb is None:
        assert depth_rgb_scale is not None
        depth, _ = align_size(depth, None, factor=depth_rgb_scale)
        height, width = depth.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, camera.intrinsic.data
        )
        depth_map = o3d.geometry.Image(np.asarray(depth, order="C"))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_map, intrinsic, camera.extrinsic.data
        )
    else:
        depth, _ = align_size(depth, rgb)
        height, width = depth.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            camera.intrinsic.fx,
            camera.intrinsic.fy,
            camera.intrinsic.cx,
            camera.intrinsic.cy,
        )

        depth_map = o3d.geometry.Image(np.asarray(depth, order="C"))
        rgb_map = o3d.geometry.Image(rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_map, depth_map, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic, camera.extrinsic.data
        )
    return PointcloudTensor.from_open3d(pcd)


def generate_pcd_from_depth_photoid(depth, camera, rgb, stride=1):
    camK = camera.intrinsic.data
    T_WH_cam = camera.extrinsic.data
    pointcloud, pointcloud_color = [], []
    for y in range(0, depth.shape[0]):
        if y % stride != 0:
            continue
        for x in range(0, depth.shape[1]):
            if x % stride != 0:
                continue
            depth_value = depth[y, x]
            if depth_value < 0.1:
                continue
            pt = np.asarray([x, y, 1], dtype=np.float64)
            bearing = np.linalg.inv(camK) @ pt
            p_rgb = bearing * depth_value
            if p_rgb[2] < 0:
                continue
            pt_xyz = T_WH_cam[:3, :3] @ p_rgb + T_WH_cam[:3, 3]
            pointcloud.append(pt_xyz)
            pointcloud_color.append(rgb[y, x])

    # assert len(pointcloud) > 0
    pcd = PointcloudTensor(pointcloud, color=pointcloud_color)
    return pcd


def generate_pcd_from_depth(
    depth: Tensor,
    camera: Camera,
    rgb: Optional[Tensor] = None,
    depth_rgb_scale: Optional[float] = 1.0,
    coordinate_type: Optional[str] = "open3d",
    stride: Optional[int] = 1,
) -> Optional[PointcloudTensor]:
    if coordinate_type == "open3d":
        return generate_pcd_from_depth_open3d(
            depth, camera, rgb, depth_rgb_scale
        )
    elif coordinate_type == "photoid":
        return generate_pcd_from_depth_photoid(depth, camera, rgb, stride)
    else:
        raise NotImplementedError


def merge_pointcloud(
    pointcloud_list: List[PointcloudTensor], quiet: bool = False
) -> PointcloudTensor:
    """Merge pointcloud_list to one PointcloudTensor."""
    if not OPEN3D_INSTALLED:
        return None
    pcd = o3d.geometry.PointCloud()

    if not quiet:
        t = tqdm.tqdm(total=len(pointcloud_list))
    for pointcloud in pointcloud_list:
        if not quiet:
            t.update()
        pcd_item = pointcloud.to_open3d()
        pcd += pcd_item
    return PointcloudTensor.from_open3d(pcd)


def align_ts(ts1: List[float], ts2: List[float]) -> List[float]:
    """Align ts2 to ts1."""
    ts1 = np.asarray(sorted(ts1))
    ts2 = np.asarray(sorted(ts2))
    ts2_aligned_index = np.abs(ts1[None, :] - ts2[:, None]).argmin(axis=0)
    return ts2[ts2_aligned_index].tolist()


def align_timestamp(ts_list: List[List[float]], index=0) -> List[Dict]:
    """Align timestamp.

    Args:
        ts_list: input timestamp list.
        index: seed timestamp index in ts_list.

    Example:
        >>> rgb_aligned, depth_aligned = align_timestamp([pose_ts, rgb_ts, depth_ts], index=0)
    """
    assert 0 <= index < len(ts_list)
    seed_ts = ts_list[index]

    aligned_ts_list: List[List] = []
    for idx, dst_ts in enumerate(ts_list):
        if idx == index:
            continue
        aligned_ts = align_ts(seed_ts, dst_ts)
        assert len(aligned_ts) == len(seed_ts)
        aligned_ts_list.append(aligned_ts)

    if len(ts_list) == 2 and len(aligned_ts_list) == 1:
        return aligned_ts_list[0]
    else:
        return aligned_ts_list


def project_depth_to_rgb(depth, rgb, K_tof, K_rgb, T_rgb_tof, stride=1):
    """Project depth to rgb."""
    depthmap = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    for y in range(0, depth.shape[0], stride):
        for x in range(0, depth.shape[1], stride):
            _d = depth[y, x]
            if _d < 0.1:
                continue
            pt = np.array([x, y, 1])
            bearing = np.linalg.inv(K_tof).dot(pt)
            # bearing /= bearing[2]
            p_tof = bearing * _d
            p_left = T_rgb_tof[:3, :3].dot(p_tof) + T_rgb_tof[:3, 3]
            if p_left[2] < 0:
                continue
            uv = K_rgb.dot((p_left / p_left[2])).astype(np.int32)
            if (
                uv[0] < 0
                or uv[1] < 0
                or uv[0] >= rgb.shape[1]
                or uv[1] >= rgb.shape[0]
            ):
                continue
            depthmap[uv[1], uv[0]] = _d
    return depthmap


def transform_points(
    points,
    scale=1.0,
    rotation=0.0,
    rotation_axis=[0.0, 0.0, 1.0],
    translation=[0.0, 0.0, 0.0],
):
    points *= scale
    rotation_matrix = trimesh.transformations.rotation_matrix(
        rotation, np.asarray(rotation_axis)
    )
    rotation_matrix = rotation_matrix[:3, :3]
    points = points @ rotation_matrix.T + np.asarray(translation)
    return points


def get_3D_corners(model_points):
    """
    Get 3d corners from cad vertices.

    Args:
        model_points: np.ndarray, [N, 3]

    Return:
        corners: np.ndarray, [9, 3]
    """
    scale = np.max(model_points, axis=0) - np.min(model_points, axis=0)
    shift = np.mean(model_points, axis=0)
    bbox_3d = (
        np.array(
            [
                [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
            ]
        )
        + shift
    )
    bbox_3d = np.concatenate(([shift], bbox_3d), axis=0)
    return bbox_3d


def load_points_3d_from_cad(
    cad_path,
    n_points=1048,
    src_unit="millimeter",
    dst_unit="meter",
    legacy=False,
):
    """Load 3d points from cad file."""
    mesh = trimesh.load_mesh(cad_path)
    load_from_glb = False
    if isinstance(mesh, trimesh.Scene):
        scene = mesh
        meshes = list(mesh.geometry.values())
        mesh = meshes[0]
        load_from_glb = True

    if n_points > 0:
        model_points = mesh.sample(n_points).astype(np.float32)
    else:
        model_points = mesh.vertices.astype(np.float32)

    if src_unit == dst_unit:
        pass
    elif src_unit == "millimeter" and dst_unit == "meter":
        model_points /= 1000.0
    elif src_unit == "meter" and dst_unit == "millimeter":
        model_points *= 1000.0
    else:
        raise ValueError(f"Unsupported unit: {src_unit}, {dst_unit}")

    if load_from_glb and legacy:
        model_points = transform_points(model_points, scale=100)
        model_points = transform_points(
            model_points, rotation=np.pi / 2, rotation_axis=[0, 1, 0]
        )
        model_points = transform_points(
            model_points, rotation=np.pi / 2, rotation_axis=[0, 0, 1]
        )
        model_points = transform_points(
            model_points, rotation=np.pi / 2, rotation_axis=[0, 1, 0]
        )
    return model_points
