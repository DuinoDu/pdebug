import json
import os
import random
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from pdebug.utils.env import (
    OPEN3D_INSTALLED,
    SCIPY_INSTALLED,
    TORCH_INSTALLED,
    TRIMESH_INSTALLED,
)
from pdebug.utils.fileio import load_yaml_with_head

import cv2
import numpy as np
import pycocotools.mask as cocomask
from dataclasses_json import dataclass_json
from imantics import Mask, Polygons

__all__ = [
    "Tensor",
    "RayTensor",
    "PointcloudTensor",
    "CameraExtrinsic",
    "CameraIntrinsic",
    "Camera",
    "x_to_ndarray",
    "NerfCamera",
    "Open3dCameraTrajectory",
    "ColmapImages",
    "ColmapPoints",
    "PoseList",
    "SamResult",
    "Segmentation",
    "SemsegResult",
    "Sam6dResult",
]


Tensor = np.ndarray


if TORCH_INSTALLED:
    import torch

    Tensor = Union[np.ndarray, torch.Tensor]

if OPEN3D_INSTALLED:
    import open3d as o3d

if SCIPY_INSTALLED:
    from scipy.spatial.transform import Rotation as sstR

if TRIMESH_INSTALLED:
    import trimesh


def x_to_ndarray(x) -> np.ndarray:
    """Convert any tensor to np.ndarray."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, list):
        x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@dataclass
class RayTensor:

    data: Tensor
    target_ndim: int = 3

    def __post_init__(self):  # type: ignore
        self.data = x_to_ndarray(self.data)
        if self.data.ndim == 3:
            self.data = self.data.squeeze()
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis]
        assert (
            self.data.ndim == self.target_ndim
        ), f"Bad data ndim ({self.data.ndim} != {self.target_ndim})"

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def ray_num(self) -> int:
        return self.data.shape[1]

    @property
    def channel_num(self) -> int:
        return self.data.shape[2]


@dataclass
class PointcloudTensor:

    data: Tensor
    color: Optional[Tensor] = None
    label: Optional[Tensor] = None
    target_ndim: int = 2

    def __post_init__(self):  # type: ignore
        self.data = x_to_ndarray(self.data).astype(np.float32)
        assert (
            self.data.ndim == self.target_ndim
        ), f"Bad data ndim ({self.data.ndim} != {self.target_ndim})"
        if self.color is not None:
            self.color = x_to_ndarray(self.color)
            assert self.data.shape[0] == self.color.shape[0]
            if self.color.max() <= 1.0:
                self.color = (x_to_ndarray(self.color) * 255).astype(np.uint8)
            else:
                self.color = x_to_ndarray(self.color).astype(np.uint8)
        if self.label is not None:
            self.label = x_to_ndarray(self.label)
            assert self.data.shape[0] == self.label.shape[0]

    @property
    def point_num(self) -> int:
        return self.data.shape[0]

    @property
    def channel_num(self) -> int:
        return self.data.shape[1]

    @property
    def dtype(self):
        return self.data.dtype

    @classmethod
    def to_homo(cls, point):
        """Convert to homo format, with shape (N, 4)."""
        is_pcd = True
        if not isinstance(point, PointcloudTensor):
            is_pcd = False
            point = PointcloudTensor(point)

        homo_data = np.hstack(
            (point.data[:, :3], np.ones((point.point_num, 1), dtype="float32"))
        )
        if is_pcd:
            return cls(homo_data)
        else:
            return homo_data

    @property
    def x(self) -> Tensor:
        return self.data[:, 0]

    @property
    def y(self) -> Tensor:
        return self.data[:, 1]

    @property
    def z(self) -> Tensor:
        return self.data[:, 2]

    @property
    def homo(self) -> Optional[Tensor]:
        assert (
            self.channel_num == 4
        ), "Please convert to homo-format by `to_home`"
        return self.data[:, 3]

    def label_as_color(self):
        from pdebug.visp.colormap import Colormap

        assert self.label is not None
        label_num = int(self.label.max() + 1)
        colormap = Colormap(label_num)
        for idx, label in enumerate(self.label):
            self.color[idx] = colormap[int(label)]
        return self

    @classmethod
    def from_colmap_point3d(cls, txtfile: str):
        """
        Initialize pcd from "points3D.txt" in colmap sparse output folder.
        """
        fid = open(txtfile, "r")
        points = []
        for line in fid:
            if line.startswith("#"):
                continue
            _, x, y, z, r, g, b, error, *args = [
                float(i) for i in line.split(" ")
            ]
            points.append([x, y, z, r, g, b])
        points = np.asarray(points, dtype=np.float32)
        fid.close()
        points_tensor = points[:, 0:3]
        colors_tensor = points[:, 3:6]
        return cls(points_tensor, color=colors_tensor)

    def to_open3d(self, savename: str = None) -> Union["PointCloud", None]:
        """Save or return open3d pcd file."""
        assert OPEN3D_INSTALLED, "`to_open3d` requires `open3d` installed."
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data[:, :3])
        if self.color is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.color)
        if savename:
            assert str(savename).endswith(
                ".pcd"
            ), "savename should endswith `.pcd`"
            o3d.io.write_point_cloud(savename, pcd, print_progress=True)
        else:
            return pcd

    @classmethod
    def from_open3d(cls, pcd_obj: Union[str, "PointCloud"]):
        assert OPEN3D_INSTALLED, "`from_open3d` requires `open3d` installed."
        if isinstance(pcd_obj, str) and os.path.exists(pcd_obj):
            pcd = o3d.io.read_point_cloud(pcd_obj)
        elif isinstance(pcd_obj, o3d.geometry.PointCloud):
            pcd = pcd_obj
        else:
            raise TypeError(f"Unknown type: {pcd_obj}({type(pcd_obj)})")
        color = np.asarray(pcd.colors) if pcd.has_colors() else None
        return cls(np.asarray(pcd.points), color=color)

    @classmethod
    def from_trimesh(cls, ply_file: str):
        assert (
            TRIMESH_INSTALLED
        ), "`from_trimesh` requires `trimesh` installed."
        mesh = trimesh.load(ply_file)
        point = np.asarray(mesh.vertices)
        color = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3]
        return cls(point, color=color)

    def to_ply(self, savename: str = None, quiet: bool = True) -> None:
        """Save to ply file."""
        assert str(savename).endswith(".ply")

        assert self.data.shape[1] == 3
        if self.color is None:
            self.color = np.ones(self.data.shape).astype(np.uint8) * 255

        # Write header of .ply file
        fid = open(savename, "wb")
        fid.write(bytes("ply\n", "utf-8"))
        fid.write(bytes("format binary_little_endian 1.0\n", "utf-8"))
        fid.write(bytes("element vertex %d\n" % self.point_num, "utf-8"))
        fid.write(bytes("property float x\n", "utf-8"))
        fid.write(bytes("property float y\n", "utf-8"))
        fid.write(bytes("property float z\n", "utf-8"))
        fid.write(bytes("property uchar red\n", "utf-8"))
        fid.write(bytes("property uchar green\n", "utf-8"))
        fid.write(bytes("property uchar blue\n", "utf-8"))
        fid.write(bytes("end_header\n", "utf-8"))

        for i in range(self.point_num):
            fid.write(
                bytearray(
                    struct.pack(
                        "fffccc",
                        self.data[i, 0],
                        self.data[i, 1],
                        self.data[i, 2],
                        # self.color[i, 0].tostring(),
                        # self.color[i, 1].tostring(),
                        # self.color[i, 2].tostring(),
                        self.color[i, 0].tobytes(),
                        self.color[i, 1].tobytes(),
                        self.color[i, 2].tobytes(),
                    )
                )
            )
            if not quiet:
                print(f"{i} / {self.point_num}")
        fid.close()
        return True

    def to_ply_ascii(self, savename: str = None, quiet: bool = True) -> None:
        """Save to ply file in ASCII format."""
        assert str(savename).endswith(".ply")

        assert self.data.shape[1] == 3
        if self.color is None:
            self.color = np.ones(self.data.shape).astype(np.uint8) * 255

        # Write header of .ply file
        fid = open(savename, "w")
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex %d\n" % self.point_num)
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar red\n")
        fid.write("property uchar green\n")
        fid.write("property uchar blue\n")
        fid.write("end_header\n")

        for i in range(self.point_num):
            fid.write(
                f"{self.data[i, 0]:.6f} {self.data[i, 1]:.6f} {self.data[i, 2]:.6f} "
                f"{int(self.color[i, 0])} {int(self.color[i, 1])} {int(self.color[i, 2])}\n"
            )
            if not quiet:
                print(f"{i} / {self.point_num}")
        fid.close()
        return True

    def to_html(self, title, size=10, scale=1, output_dir="show_pcd"):
        from pdebug.utils.three_utils import threejs_pcd
        from pdebug.utils.web_engine import ThreeEngine

        js_code = threejs_pcd(self, size=size)
        title = title.replace(" ", "__")
        engine = ThreeEngine("simple", res_dir=output_dir + f"/{title}")
        engine.render(js_code=js_code, scale=scale, title=title)

    def __add__(self, item: "PointcloudTensor"):
        data = np.concatenate((self.data, item.data), axis=0)
        if item.color is not None and self.color is not None:
            color = np.concatenate((self.color, item.color), axis=0)
        else:
            color = None
        if item.label is not None and self.label is not None:
            label = np.concatenate((self.label, item.label), axis=0)
        else:
            label = None
        return PointcloudTensor(data, color, label)

    def __iadd__(self, item: "PointcloudTensor"):
        self.data = np.concatenate((self.data, item.data), axis=0)
        if item.color is not None and self.color is not None:
            self.color = np.concatenate((self.color, item.color), axis=0)
        if item.label is not None and self.label is not None:
            self.label = np.concatenate((self.label, item.label), axis=0)
        return self

    def reduce_point_num(self, target_point_num):
        index = list(range(0, self.point_num))
        random.shuffle(index)
        index = index[:target_point_num]
        self.data = self.data[index]
        if self.color is not None:
            self.color = self.color[index]
        if self.label is not None:
            self.label = self.label[index]
        assert self.point_num == target_point_num

    def fill_color(self, color: List[int]):
        if self.color is None:
            self.color = np.zeros((self.data.shape[0], 3), dtype=np.uint8)
            for i in range(3):
                self.color[:, i] = color[i]
        return self

    def transform(self, transform_matrix: Tensor) -> "PointcloudTensor":
        """
        Apply 4x4 transformation matrix to the point cloud.

        Args:
            transform_matrix: 4x4 transformation matrix

        Returns:
            PointcloudTensor: New transformed point cloud
        """
        transform_matrix = x_to_ndarray(transform_matrix)
        assert transform_matrix.shape == (
            4,
            4,
        ), f"Expected 4x4 matrix, got {transform_matrix.shape}"

        # Convert points to homogeneous coordinates
        homo_points = self.to_homo(self).data

        # Apply transformation
        transformed_points = (transform_matrix @ homo_points.T).T

        # Convert back to 3D coordinates by dividing by homogeneous coordinate
        transformed_points = (
            transformed_points[:, :3] / transformed_points[:, 3:4]
        )

        return PointcloudTensor(
            transformed_points, color=self.color, label=self.label
        )

    def copy(self):
        data = self.data.copy()
        color = self.color.copy() if self.color is not None else None
        label = self.label.copy() if self.label is not None else None
        return PointcloudTensor(data, color, label)


@dataclass
class CameraExtrinsic:
    """Camera extrinsic matrix.

    [r00, r01, r02, x]
    [r10, r11, r12, y]
    [r20, r21, r22, z]
    [  0,   0,   0, 1]

    """

    data: Tensor
    target_ndim: int = 2
    target_shape: Tuple[int, int] = (4, 4)

    def __post_init__(self):  # type: ignore
        self.data = x_to_ndarray(self.data)
        if self.data.ndim == 3:
            if self.data.shape[0] == 1:
                self.data = self.data[0]
            else:
                raise NotImplementedError(
                    "Found multiple extrinsic matrix, not implemented"
                )
        assert self.data.ndim == self.target_ndim
        self.data = CameraExtrinsic.reformat(self.data)
        assert (
            self.data.shape == self.target_shape
        ), f"shape mismatch: {self.data.shape} != {self.target_shape}"

    @staticmethod
    def reformat(data):
        """Format matrix, 3x4 or 4x3 to 4x4."""
        if data.shape == (4, 4):
            return data
        if data.shape == (4, 3):
            data = np.transpose(data, (1, 0))
        if data.shape == (3, 4):
            data_4x4 = np.zeros((4, 4), dtype=data.dtype)
            data_4x4[:3] = data
            data_4x4[3] = [0.0, 0.0, 0.0, 1.0]
        else:
            raise NotImplementedError(
                f"data shape ({data.shape}) is not supported"
            )
        return data_4x4

    @property
    def rotation(self) -> Tensor:
        return self.data[:3, :3]

    def as_euler(self, order="zyx", degrees=True) -> Tensor:
        return sstR.from_matrix(self.R).as_euler(order, degrees=degrees)

    def as_quat(self) -> Tensor:
        """Return quaternion in xyzw order."""
        return sstR.from_matrix(self.R).as_quat()

    @property
    def translation(self) -> Tensor:
        return self.data[:3, 3]

    @property
    def xyz(self) -> Tensor:
        return self.translation

    @property
    def t(self) -> Tensor:
        return self.translation

    @property
    def R(self) -> Tensor:
        return self.rotation

    @classmethod
    def fromRt(cls, rotation, position):
        """
        Create CameraExtrinsic from rotation and position.

        Args:
            rotation: rotation matrix or quaternion in xyzw order.
            position: camera position in world.
        """
        assert SCIPY_INSTALLED, "Please install scipy."
        position = x_to_ndarray(position)
        assert position.shape == (3,)

        rotation = x_to_ndarray(rotation)
        if rotation.ndim == 1 and rotation.size == 4:
            rotation = sstR.from_quat(rotation).as_matrix()
        assert rotation.ndim == 2 and rotation.shape == (
            3,
            3,
        ), f"Unvalid rotation format: {rotation}"

        data = np.eye(4)
        data[:3, :3] = rotation
        data[:3, 3] = position
        return cls(data)

    @classmethod
    def from_colmap_image(cls, pose_info):
        tx, ty, tz = pose_info["TX"], pose_info["TY"], pose_info["TZ"]
        qx, qy, qz, qw = (
            pose_info["QX"],
            pose_info["QY"],
            pose_info["QZ"],
            pose_info["QW"],
        )
        rotation = np.asarray([qx, qy, qz, qw])
        position = np.asarray([tx, ty, tz])
        return cls.fromRt(rotation, position)


Pose = CameraExtrinsic


@dataclass
class CameraIntrinsic:
    """Camera intrinsic matrix.

    [fx,  0,  cx]
    [ 0, fy,  cy]
    [ 0,  0,  1]

    """

    data: Tensor
    target_ndim: int = 2
    target_shape: Tuple[int, int] = (3, 3)
    width: Optional[int] = None
    height: Optional[int] = None
    camera_model: Optional[str] = "pinhole"
    distortion_model: Optional[str] = None
    distortion_coefficients: Optional[Tensor] = None
    xi: Optional[float] = None

    def __post_init__(self):  # type: ignore
        self.data = x_to_ndarray(self.data).astype(np.float64)
        if len(self.data) == 4:
            fx, fy, cx, cy = self.data
            self.data = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ]
            )
        if self.data.ndim == 3:
            if self.data.shape[0] == 1:
                self.data = self.data[0]
            else:
                raise NotImplementedError(
                    "Found multiple intrinsic matrix, not implemented"
                )
        assert self.data.ndim == self.target_ndim
        assert (
            self.data.shape == self.target_shape
        ), f"shape mismatch: {self.data.shape} != {self.target_shape}"
        if self.distortion_coefficients is not None:
            self.distortion_coefficients = np.asarray(
                self.distortion_coefficients
            )

    @property
    def fx(self):
        return self.data[0, 0]

    @property
    def fy(self):
        return self.data[1, 1]

    @property
    def cx(self):
        return self.data[0, 2]

    @property
    def cy(self):
        return self.data[1, 2]

    @property
    def w(self):
        return self.cx * 2 if self.width is None else self.width

    @property
    def h(self):
        return self.cy * 2 if self.height is None else self.height

    def tolist(self) -> List[float]:
        return [self.fx, self.fy, self.cx, self.cy]

    def to_open3d(
        self, width=None, height=None, savename=None
    ) -> "o3d.camera.PinholeCameraIntrinsic":
        assert OPEN3D_INSTALLED, "`to_open3d` requires `open3d` installed."
        if not width:
            width = self.w
        if not height:
            height = self.h
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            int(width), int(height), self.fx, self.fy, self.cx, self.cy
        )
        if savename:
            assert savename.endswith(".json")
            o3d.io.write_pinhole_camera_intrinsic(savename, intrinsic_o3d)
        return intrinsic_o3d

    @classmethod
    def from_colmap_txt(cls, txtfile):
        """Create camera intrinsic from colmap camera txt."""
        lines = [
            l.strip()
            for l in open(txtfile, "r").readlines()
            if not l.startswith("#")
        ]
        if len(lines) > 1:
            raise NotImplementedError(f"multiple cameras found in {txtfile}")
        CAMERA_ID, MODEL, WIDTH, HEIGHT, *PARAMS = lines[0].split(" ")
        if MODEL == "PINHOLE":
            assert len(PARAMS)
            fx, fy, cx, cy = [float(x) for x in PARAMS]
            distortion_coefficients = None
        elif MODEL == "OPENCV":
            fx, fy, cx, cy = [float(x) for x in PARAMS[:4]]
            # not test
            distortion_coefficients = [float(x) for x in PARAMS[4:]]
        else:
            raise NotImplementedError
        return cls(
            [fx, fy, cx, cy],
            width=int(WIDTH),
            height=int(HEIGHT),
            camera_model=MODEL,
            distortion_coefficients=distortion_coefficients,
        )

    def resize(self, factor):
        self.data[:2, :] *= factor


@dataclass
class Camera:

    extrinsic: CameraExtrinsic
    intrinsic: CameraIntrinsic

    def __post_init__(self):
        if not isinstance(self.extrinsic, CameraExtrinsic):
            self.extrinsic = CameraExtrinsic(self.extrinsic)
        if not isinstance(self.intrinsic, CameraIntrinsic):
            self.intrinsic = CameraIntrinsic(self.intrinsic)

    @property
    def E(self):
        return self.extrinsic

    @property
    def I(self):
        return self.intrinsic

    def project_point_to_image(
        self,
        point: Union[Tensor, PointcloudTensor],
        image: Optional[Tensor] = None,
        radius: int = 3,
    ) -> Tensor:
        """Project point to image.

        In details, it do three steps:
          1. transfrom points from world space to camera space.
          2. project point in camera space to camera plane.
          3. draw point on image.
        """
        if not isinstance(point, PointcloudTensor):
            point = PointcloudTensor(point)
        assert point.point_num, "Empty point!"

        point_camera = Camera.world_to_camera(point, self.extrinsic)
        point_image = Camera.camera_to_image(point_camera, self.intrinsic)

        if image is None:
            image = np.zeros((int(self.I.h), int(self.I.w), 3), dtype=np.uint8)
        else:
            image = image.astype(np.uint8)

        # For colmap camera pose, it's not always true.
        # assert image.shape == (self.I.h, self.I.w, 3)

        image = draw_point(image, point_image, opacity=0.5, radius=radius)
        return image

    def capture(self, point: Tensor) -> Tensor:
        if OPEN3D_INSTALLED and isinstance(point, o3d.geometry.PointCloud):
            point = PointcloudTensor.from_open3d(point)
        assert isinstance(point, (PointcloudTensor, np.ndarray))
        return self.project_point_to_image(point)

    @staticmethod
    def world_to_camera(
        point: PointcloudTensor,
        extrinsic: CameraExtrinsic,
    ) -> PointcloudTensor:
        """Transform point from world space to camera space."""
        homo_points = PointcloudTensor.to_homo(point)
        T_S_W = np.linalg.inv(extrinsic.data)
        point_camera = np.dot(homo_points.data, T_S_W.T)
        point_camera /= point_camera[:, -1].reshape((-1, 1))
        return PointcloudTensor(point_camera[:, :3], color=point.color)

    @staticmethod
    def camera_to_image(
        point: PointcloudTensor,
        intrinsic: CameraIntrinsic,
        dist_coeffs: Optional[Tensor] = None,
    ) -> PointcloudTensor:
        """Project point in camera space to image space."""
        is_pcd = True
        if not isinstance(point, PointcloudTensor):
            is_pcd = False
            point = PointcloudTensor(point)

        point.data = point.data.astype(np.float32)
        intrinsic.data = intrinsic.data.astype(point.dtype)
        assert (
            point.channel_num == 3
        ), f"Require point channel is 3, but found {point.channel_num}"

        if dist_coeffs:
            raise NotImplementedError
        dist_coeffs = np.zeros([4, 1], point.dtype)

        # # remove z < 0 value
        # point = point[point[:, 2] > 0, :3].astype(point.float32)

        r_vec, _ = cv2.Rodrigues(np.identity(3, point.dtype))
        t_vec = np.zeros(shape=(3, 1), dtype=point.dtype)

        point_image = cv2.projectPoints(
            point.data, r_vec, t_vec, intrinsic.data, dist_coeffs
        )[0]
        point_image = np.asarray(point_image).squeeze()  # [N, 2]
        if point_image.ndim == 1:
            point_image = point_image[np.newaxis, :]

        if is_pcd:
            return PointcloudTensor(point_image, color=point.color)
        else:
            return point_image

    def project_point_to_image_v2(
        self,
        point: Union[Tensor, PointcloudTensor],
        image: Optional[Tensor] = None,
        color: Tuple[int] = (255, 255, 0),
        radius: int = 3,
        return_uv: bool = False,
    ) -> Tensor:
        """Project point to image, without distortion process.

        In details, it do three steps:
          1. transfrom points from world space to camera space.
          2. project point in camera space to camera plane.
          3. draw point on image.
        """
        if not isinstance(point, PointcloudTensor):
            point = PointcloudTensor(point)
        assert point.point_num, "Empty point!"

        if image is None:
            image = np.zeros((int(self.I.h), int(self.I.w), 3), dtype=np.uint8)
        else:
            image = image.astype(np.uint8).copy()

        if return_uv:
            uv_list = []

        tsw = self.extrinsic.data
        K_rgb = self.intrinsic.data
        for xyz in point.data:
            p_rgb = tsw[:3, :3].dot(xyz) + tsw[:3, 3]
            if p_rgb[2] < 0:
                continue
            uv = K_rgb.dot((p_rgb / p_rgb[2])).astype(np.int32)
            if return_uv:
                uv_list.append(uv[:2])
            if (
                uv[0] < 0
                or uv[1] < 0
                or uv[0] >= image.shape[1]
                or uv[1] >= image.shape[0]
            ):
                continue
            image = cv2.circle(image, uv[:2], radius, color, 2)

        if return_uv:
            return image, np.asarray(uv_list)
        else:
            return image

    @classmethod
    def from_nerf_camera(
        cls,
        nerf_camera,
        *,
        use_half_wh_as_cxcy: bool = False,
    ) -> List["Camera"]:
        """Initialize Camera list from nerf_camera."""
        fx = nerf_camera._data["fl_x"]
        fy = nerf_camera._data["fl_y"]
        cx = nerf_camera._data["cx"]
        cy = nerf_camera._data["cy"]
        if use_half_wh_as_cxcy and "w" in nerf_camera._data:
            cx = nerf_camera._data["w"] // 2
            cy = nerf_camera._data["h"] // 2
        intrinsic = [fx, fy, cx, cy]

        cameras = []
        for frame in nerf_camera.frames:
            extrinsic = np.asarray(frame["transform_matrix"])
            camera = cls(extrinsic, intrinsic)
            cameras.append(camera)
        return cameras

    @classmethod
    def from_yamlfile(
        cls,
        yaml_file: str,
        version: Optional[str] = "v1",
    ) -> "Camera":
        """Create Camera from yaml file."""
        assert version == "v1"
        config = load_yaml_with_head(yaml_file)
        if config["camera_model"] == "omni":
            xi, fx, fy, cx, cy = config["intrinsics"]
        elif config["camera_model"] == "pinhole":
            fx, fy, cx, cy = config["intrinsics"]
            xi = 0
        else:
            raise NotImplementedError
        w, h = config["resolution"]
        intrinsic = [fx, fy, cx, cy]

        camera_intrinsic = CameraIntrinsic(
            intrinsic,
            xi=xi,
            width=w,
            height=h,
            camera_model=config["camera_model"],
            distortion_model=config["distortion_model"],
            distortion_coefficients=config["distortion_coefficients"],
        )
        extrinsic = np.asarray(config["T_BS"]["data"]).reshape(4, 4)
        camera_extrinsic = CameraExtrinsic(extrinsic)
        return cls(camera_extrinsic, camera_intrinsic)


def draw_point(
    image: Tensor,
    point: PointcloudTensor,
    *,
    color: Tuple[int] = (255, 255, 0),
    radius: int = 1,
    opacity: float = 0.0,
) -> Tensor:
    """Draw point on image."""
    image = image.copy()
    if not isinstance(point, PointcloudTensor):
        point = PointcloudTensor(point)
    assert point.channel_num == 2, "Only support 2-D point."

    # remove image outside point.
    indicator = np.logical_and.reduce(
        (
            0 <= point.x,
            0 <= point.y,
            point.x < image.shape[1],
            point.y < image.shape[0],
        )
    )
    point.data = point.data[indicator]
    if point.color is not None:
        point.color = point.color[indicator]

    if point.point_num == 0:
        return image

    # convert float to int
    point.data = (point.data + 0.5).astype("int32")

    if point.color is not None:
        color = point.color.astype(int).tolist()
    else:
        color = [color for _ in range(point.point_num)]

    if opacity:
        assert 0 < opacity <= 1.0
        front = np.zeros_like(image)
        for (x, y, c) in zip(point.x, point.y, color):
            cv2.circle(front, (x, y), radius, tuple(c), -1)
        image = out = cv2.addWeighted(image, 1.0, front, opacity, 1)
    else:
        for (x, y, c) in zip(point.x, point.y, color):
            cv2.circle(image, (x, y), radius, tuple(c), -1)
    return image


class NerfCamera:

    _KEYS = [
        "camera_angle_x",
        "camera_angle_y",
        "fl_x",
        "fl_y",
        "k1",
        "k2",
        "p1",
        "p2",
        "cx",
        "cy",
        "w",
        "h",
        "aabb_scale",
    ]

    def __init__(self, data: Dict, sort_key="file_path"):
        self._data = data
        assert "frames" in self._data
        self._frames = self._data["frames"]
        if sort_key:
            self._frames = sorted(self._frames, key=lambda x: x["file_path"])

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        elif name in self._KEYS:
            raise AttributeError("attr name: {name} not found in json file.")

    def __len__(self):
        return len(self._frames)

    @classmethod
    def fromfile(cls, json_file: str):
        with open(json_file, "r") as fid:
            data = json.load(fid)
        return cls(data)

    @property
    def frames(self):
        return self._frames

    def dump(self, output: str):
        """Dump to json file."""
        with open(output, "w") as fid:
            json.dump(self._data, fid, indent=4)

    @classmethod
    def from_o3d_trajectory(cls, o3d_camera: "Open3dCameraTrajectory"):
        """Initialize nerf camera transform from open3d camera trajectory."""

        intrinsic = o3d_camera.parameters[0]["intrinsic"]
        data = {
            "fl_x": intrinsic["intrinsic_matrix"][0],
            "fl_y": intrinsic["intrinsic_matrix"][4],
            "cx": intrinsic["intrinsic_matrix"][6],
            "cy": intrinsic["intrinsic_matrix"][7],
            "w": intrinsic["width"],
            "h": intrinsic["height"],
            "frames": [],
        }
        for p in o3d_camera.parameters:
            extrinsic = np.asarray(p["extrinsic"]).reshape(4, 4).transpose()
            item = {"file_path": None, "transform_matrix": extrinsic.tolist()}
            data["frames"].append(item)
        return cls(data, sort_key=None)

    @staticmethod
    def is_valid(json_file: str) -> bool:
        """If input json_file is a valid nerf camera pose file."""
        with open(json_file, "r") as fid:
            data = json.load(fid)
        valid = (
            "frames" in data
            and isinstance(data["frames"], list)
            and "transform_matrix" in data["frames"][0]
        )
        return valid

    def align_first_pose_to_origin(self) -> np.ndarray:
        """Align first pose to origin point."""
        p0 = self.frames[0]["transform_matrix"]
        t_mat = np.array(
            [
                [1, 0, 0, -p0[0][3]],
                [0, 1, 0, -p0[1][3]],
                [0, 0, 1, -p0[2][3]],
                [0, 0, 0, 1],
            ]
        )
        for f in self.frames:
            f["transform_matrix"] = np.matmul(
                t_mat, f["transform_matrix"]
            ).tolist()
        return t_mat


class Open3dCameraTrajectory:
    def __init__(self, data: Dict):
        self._data = data
        assert "parameters" in data
        self._parameters = data["parameters"]

    def dump(self, output: str):
        """Dump to json file."""
        with open(output, "w") as fid:
            json.dump(self._data, fid, indent=4)

    @classmethod
    def from_nerf(cls, camera: NerfCamera):
        """Initialize open3d camera trajectory from nerf camera transform."""

        data = {
            "class_name": "PinholeCameraTrajectory",
            "parameters": [],
            "version_major": 1,
            "version_minor": 0,
        }
        fx = camera.fl_x
        fy = camera.fl_y
        cx = camera.cx
        cy = camera.cy
        intrinsic = {
            "height": int(camera.h),
            "width": int(camera.w),
            "intrinsic_matrix": [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0],
        }
        for frame in camera.frames:
            extrinsic = np.asarray(frame["transform_matrix"])
            extrinsic = extrinsic.transpose().flatten().tolist()
            item = {
                "class_name": "PinholeCameraParameters",
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "version_major": 1,
                "version_minor": 0,
            }
            data["parameters"].append(item)
        return cls(data)

    @classmethod
    def fromfile(cls, json_file: str):
        with open(json_file, "r") as fid:
            data = json.load(fid)
        return cls(data)

    def __len__(self):
        return len(self._parameters)

    @property
    def parameters(self):
        return self._parameters

    @staticmethod
    def is_valid(json_file: str) -> bool:
        """If input json_file is a valid open3d camera trajectory file."""
        with open(json_file, "r") as fid:
            data = json.load(fid)
        valid = (
            "parameters" in data
            and isinstance(data["parameters"], list)
            and "extrinsic" in data["parameters"][0]
        )
        return valid


@dataclass
class ColmapImages:
    """
    https://colmap.github.io/format.html#images-txt

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 2, mean observations per image: 2
    """

    data: Dict[int, Dict]

    def __len__(self):
        return len(self.data)

    def get_by_image_name(self, image_name: str):
        image_name = os.path.basename(image_name)
        image_id = self.name_to_id(image_name)
        return self.data[image_id]

    read = get_by_image_name

    def name_to_id(self, name: str) -> str:
        """Convert image name to image id."""
        if not hasattr(self, "_name_to_id"):
            self._name_to_id = {
                item["IMAGE_NAME"]: key for key, item in self.data.items()
            }
        return self._name_to_id[name]

    @classmethod
    def fromfile(cls, txtfile: str):
        data = {}
        with open(txtfile, "r") as fid:
            odd_line = False
            for line in fid:
                if line.startswith("#"):
                    continue
                if not odd_line:
                    (
                        IMAGE_ID,
                        QW,
                        QX,
                        QY,
                        QZ,
                        TX,
                        TY,
                        TZ,
                        CAMERA_ID,
                        NAME,
                    ) = line.strip().split(" ")
                    data[IMAGE_ID] = {
                        "QW": float(QW),
                        "QX": float(QX),
                        "QY": float(QY),
                        "QZ": float(QZ),
                        "TX": float(TX),
                        "TY": float(TY),
                        "TZ": float(TZ),
                        "CAMERA_ID": CAMERA_ID,
                        "IMAGE_NAME": NAME,
                    }
                    odd_line = True
                else:
                    item = np.asarray(
                        [float(x) for x in line.strip().split(" ")]
                    )
                    X = item.reshape(-1, 3)[:, 0]
                    Y = item.reshape(-1, 3)[:, 1]
                    POINT3D_ID = item.reshape(-1, 3)[:, 2].astype(np.int64)
                    data[IMAGE_ID]["X"] = X
                    data[IMAGE_ID]["Y"] = Y
                    data[IMAGE_ID]["POINT3D_ID"] = POINT3D_ID
                    odd_line = False
        return cls(data)


@dataclass
class ColmapPoints:
    """
    https://colmap.github.io/format.html#points3d-txt

    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334

    """

    data: Dict[int, Dict]

    def __len__(self):
        return len(self.data)

    @classmethod
    def fromfile(cls, txtfile: str):
        data = {}
        with open(txtfile, "r") as fid:
            for line in fid:
                if line.startswith("#"):
                    continue
                (
                    POINT3D_ID,
                    X,
                    Y,
                    Z,
                    R,
                    G,
                    B,
                    ERROR,
                    *TRACK,
                ) = line.strip().split(" ")
                data[POINT3D_ID] = {
                    "XYZ": [float(X), float(Y), float(Z)],
                    "RGB": [int(R), int(G), int(B)],
                    "ERROR": float(ERROR),
                    "TRACK": TRACK,
                }
        return cls(data)

    def get_by_point_id(self, point_id) -> List[Dict]:
        """Get point by point id. If not found, set None in return list."""
        if isinstance(point_id, (int, np.int64)):
            point_id = [str(point_id)]
        ret = [self.data.get(str(_id), None) for _id in point_id]
        return ret


@dataclass
class PoseList:

    data: List[Pose]
    timestamps: List[float] = None

    @classmethod
    def from_xmlfile(cls, xmlfile, verbose: bool = False):
        """
        Initialize PoseList from Pose.xml
        """
        pose_list = []
        ts_list = []
        if verbose:
            ind = 0
            start = time.time()
            print(f"loading {xmlfile}")
        for line in open(xmlfile):
            if verbose and ind % 100000 == 0:
                print(ind)
            items = line.strip().split("'")[1::2]
            (
                x,
                y,
                z,
                rw,
                rx,
                ry,
                rz,
                timestamp,
                vx,
                vy,
                vz,
                ax,
                ay,
                az,
                wx,
                wy,
                wz,
                w_ax,
                w_ay,
                w_az,
            ) = items
            rotation = [rx, ry, rz, rw]
            position = [x, y, z]
            try:
                pose = Pose.fromRt(rotation, position)
            except ValueError as e:
                print(f"rotation: {rotation}, position: {position}")
                print(e)
                continue
            pose_list.append(pose)
            ts_list.append(float(timestamp))
            if verbose:
                ind += 1

        if verbose:
            end = time.time()
            print(f"cost {end - start} sec")

        return cls(data=pose_list, timestamps=ts_list)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        _str = f"""
        pose length: {len(self.data)}
            {self.data[0]}
            ...
        """
        return _str


@dataclass_json
@dataclass
class SamResult:
    image_path: str
    masks: np.ndarray  # [N, H, W]
    area: np.ndarray
    predicted_iou: np.ndarray
    stability_score: np.ndarray
    boxes: np.ndarray


@dataclass_json
@dataclass
class Segmentation:
    """
    Segmentation data type, in contour format.
    """

    boxes: np.ndarray
    category_id: List[int]  # 1-based
    segmentation: List[List[int]]

    @classmethod
    def from_mask(cls, mask, min_num_points=3, smooth_factor=0.0002):
        semantic_img = mask
        semantic_ids = np.unique(semantic_img)
        semantic_map = {key: semantic_img == key for key in semantic_ids}

        boxes = []
        category_id = []
        segmentation = []

        for semantic_id, semantic_mask in semantic_map.items():
            polygons = mask_to_polygons(
                semantic_mask, smooth_factor=smooth_factor
            )

            for points in polygons.points:
                num_points = points.shape[0]
                if num_points < min_num_points:
                    continue
                pa = np.asarray(points).reshape(-1, 2)
                x1, y1 = pa[:, 0].min(), pa[:, 1].min()
                x2, y2 = pa[:, 0].max(), pa[:, 1].max()
                boxes.append(np.asarray([x1, y1, x2, y2]))
                category_id.append(
                    int(semantic_id + 1)
                )  # semantic_id to category_id
                segmentation.append(
                    [points.reshape(-1).astype(np.int32).tolist()]
                )  # why wrapped in extra list?
        boxes = np.asarray(boxes)
        return cls(boxes, category_id, segmentation)

    @staticmethod
    def to_mask(
        segmentation,
        category_id,
        image_height,
        image_width,
        min_area: float = None,
    ):

        # # change 1-based to 0-based
        # cat_ids = [cat_id - 1 for cat_id in category_id]
        cat_ids = category_id

        # change 34 to -1 (others) to be in background, used below.
        cat_ids = list(map(lambda x: -1 if x == 34 else x, cat_ids))

        data = [(segm, cat_id) for segm, cat_id in zip(segmentation, cat_ids)]
        # sort by cat_id, -1 is others, 0 is VOID, 1 is ceiling ...
        data = sorted(data, key=lambda x: x[1])

        h, w = image_height, image_width
        mask_merged = np.zeros((h, w), dtype=np.uint8)

        for item in data:
            points, cat_id = item
            if cat_id == 0:  # skip VOID, leave to zero
                continue
            if np.asarray(points).ndim == 1:
                points = [points]
            # recover -1 to 34 (others)
            semantic_id = 34 if cat_id == -1 else cat_id
            polygons = Polygons(points)
            mask = polygons.mask(w, h).array
            if min_area and mask.sum() < min_area:
                continue
            mask_merged[mask != 0] = semantic_id
        return mask_merged


def mask_to_polygons(mask, smooth_factor=0.0002):
    if isinstance(mask, Mask):
        mask = mask.array
    mask = mask.astype(np.uint8)
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    polygons = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1)
    )
    polygons = polygons[0] if len(polygons) == 2 else polygons[1]

    smooth_polygons = []
    for poly in polygons:
        peri = cv2.arcLength(poly, True)
        smooth_poly = cv2.approxPolyDP(poly, smooth_factor * peri, True)
        smooth_polygons.append(smooth_poly)
    polygons = smooth_polygons

    polygons = [polygon.flatten() for polygon in polygons]
    return Polygons(polygons)


@dataclass_json
@dataclass
class SemsegResult:
    image_path: str
    masks: np.ndarray = None  # [H, W]
    segmentation: Segmentation = None
    image_width: int = None
    image_height: int = None

    def to_roi(
        self,
        min_num_points: int = 3,
        smooth_factor: float = 0.0002,
        add_image_file: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert SemsegResult to roidb (coco-roi).

        Args:
            min_num_points:
            smooth_factor:
            add_image_file: add `image_file` to roi dict.
        """
        from pdebug.piata import Input

        image_path = self.image_path
        if self.image_height is not None or self.image_width is not None:
            image_h, image_w = self.image_height, self.image_width
        elif os.path.exists(image_path):
            img = cv2.imread(image_path)
            image_h, image_w = img.shape[:2]
        else:
            raise RuntimeError("image_height and image_width must be given")

        assert self.masks is not None or self.segmentation is not None
        if self.segmentation is None:
            self.segmentation = Segmentation.from_mask(
                self.masks,
                min_num_points=min_num_points,
                smooth_polygons=smooth_factor,
            )

        roi = dict()
        roi["image_name"] = os.path.basename(image_path)
        roi["image_width"], roi["image_height"] = image_w, image_h
        roi["boxes"] = self.segmentation.boxes
        roi["category_id"] = self.segmentation.category_id
        roi["segmentation"] = self.segmentation.segmentation
        if add_image_file:
            roi["image_file"] = image_path
        return roi


@dataclass_json
@dataclass
class Sam6dResult:
    """
    Sam6dResult class for reading SAM-6D detection results and providing mask access.

    This class reads detection_ism.json and detection_pem.json files from the
    sam6d_results folder and provides interfaces to access semantic segmentation masks.
    """

    def __init__(
        self,
        results_path: str,
        det_score_thresh: float = 0.2,
        det_score_topk: int = 2,
    ):
        """
        Initialize Sam6dResult with the path to sam6d_results folder.

        Args:
            results_path: Path to the sam6d_results folder. If None, uses current directory.
        """
        self.results_path = results_path
        self.ism_data = None
        self.pem_data = None
        self.det_score_thresh = det_score_thresh
        self.det_score_topk = det_score_topk
        self._load_detection_files()

    def _load_detection_files(self):
        """Load detection_ism.json and detection_pem.json files."""
        ism_path = os.path.join(self.results_path, "detection_ism.json")
        pem_path = os.path.join(self.results_path, "detection_pem.json")

        try:
            with open(ism_path, "r") as f:
                self.ism_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ISM detection file not found: {ism_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ISM file: {e}, {ism_path}")

        # filter on ism_data
        dets = sorted(
            self.ism_data, key=lambda input: input["score"], reverse=True
        )
        dets = [d for d in self.ism_data if d["score"] > self.det_score_thresh]
        if self.det_score_topk > 0:
            sorted_scores = sorted([d["score"] for d in dets], reverse=True)[
                : self.det_score_topk
            ]
            dets = [d for d in dets if d["score"] in sorted_scores]
        self.ism_data = dets

        if os.path.exists(pem_path):
            with open(pem_path, "r") as f:
                try:
                    self.pem_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    pass

    def _decode_rle_mask(self, rle_data: Dict[str, Any]) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoding) mask to numpy array.

        Args:
            rle_data: Dictionary containing 'counts' and 'size' keys

        Returns:
            Binary mask as numpy array with shape (height, width)
        """
        counts = rle_data["counts"]
        h, w = rle_data["size"]

        try:
            rle = cocomask.frPyObjects(rle_data, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        return mask

    def get_mask(
        self, detection_index: int = 0, mask_type: str = "pem"
    ) -> np.ndarray:
        """
        Get semantic segmentation mask for a specific detection.

        Args:
            detection_index: Index of the detection to get mask for (default: 0)
            mask_type: Type of mask to return, either 'ism' or 'pem' (default: 'ism')

        Returns:
            Binary mask as numpy array with shape (height, width)

        Raises:
            ValueError: If mask_type is not 'ism' or 'pem', or if detection_index is out of range
        """
        if mask_type not in ["ism", "pem"]:
            raise ValueError("mask_type must be either 'ism' or 'pem'")

        data = self.ism_data if mask_type == "ism" else self.pem_data

        if detection_index < 0 or detection_index >= len(data):
            raise ValueError(
                f"detection_index {detection_index} is out of range. "
                f"Valid range: 0-{len(data)-1}"
            )

        detection = data[detection_index]

        if "segmentation" not in detection:
            raise ValueError(
                f"No segmentation data found for detection {detection_index}"
            )

        return self._decode_rle_mask(detection["segmentation"])

    def get_masks(self, mask_type: str = "pem") -> List[np.ndarray]:
        """
        Get all semantic segmentation masks.

        Args:
            mask_type: Type of masks to return, either 'ism' or 'pem' (default: 'ism')

        Returns:
            List of binary masks as numpy arrays
        """
        if mask_type not in ["ism", "pem"]:
            raise ValueError("mask_type must be either 'ism' or 'pem'")

        data = self.ism_data if mask_type == "ism" else self.pem_data
        masks = []

        data = sorted(data, key=lambda input: input["score"], reverse=True)
        for detection in data:
            if "segmentation" in detection:
                masks.append(self._decode_rle_mask(detection["segmentation"]))

        return masks

    def get_detection_info(
        self, detection_index: int = 0, info_type: str = "ism"
    ) -> Dict[str, Any]:
        """
        Get detection information (bbox, score, etc.) for a specific detection.

        Args:
            detection_index: Index of the detection (default: 0)
            info_type: Type of detection info, either 'ism' or 'pem' (default: 'ism')

        Returns:
            Dictionary containing detection information
        """
        if info_type not in ["ism", "pem"]:
            raise ValueError("info_type must be either 'ism' or 'pem'")

        data = self.ism_data if info_type == "ism" else self.pem_data

        if detection_index < 0 or detection_index >= len(data):
            raise ValueError(
                f"detection_index {detection_index} is out of range. "
                f"Valid range: 0-{len(data)-1}"
            )

        return data[detection_index]

    def get_all_detections(
        self, info_type: str = "ism"
    ) -> List[Dict[str, Any]]:
        """
        Get all detection information.

        Args:
            info_type: Type of detection info, either 'ism' or 'pem' (default: 'ism')

        Returns:
            List of dictionaries containing all detection information
        """
        if info_type not in ["ism", "pem"]:
            raise ValueError("info_type must be either 'ism' or 'pem'")

        return self.ism_data if info_type == "ism" else self.pem_data

    def get_image_size(self, mask_type: str = "ism") -> Tuple[int, int]:
        """
        Get the image size (height, width) from the segmentation data.

        Args:
            mask_type: Type of mask data to use, either 'ism' or 'pem' (default: 'ism')

        Returns:
            Tuple of (height, width)
        """
        if mask_type not in ["ism", "pem"]:
            raise ValueError("mask_type must be either 'ism' or 'pem'")

        data = self.ism_data if mask_type == "ism" else self.pem_data

        if not data or "segmentation" not in data[0]:
            raise ValueError(
                "No segmentation data available to determine image size"
            )

        return tuple(data[0]["segmentation"]["size"])

    def __len__(self) -> int:
        """Return the number of ISM detections."""
        return len(self.ism_data) if self.ism_data else 0

    def __repr__(self) -> str:
        """Return string representation of the Sam6dResult object."""
        return f"Sam6dResult(ism_detections={len(self.ism_data)}, pem_detections={len(self.pem_data)})"
