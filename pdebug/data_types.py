import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from pdebug.utils.env import OPEN3D_INSTALLED, TORCH_INSTALLED

import cv2
import numpy as np

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
]


Tensor = np.ndarray


if TORCH_INSTALLED:
    import torch

    Tensor = Union[np.ndarray, torch.Tensor]

if OPEN3D_INSTALLED:
    import open3d as o3d


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
    target_ndim: int = 2

    def __post_init__(self):  # type: ignore
        self.data = x_to_ndarray(self.data)
        assert (
            self.data.ndim == self.target_ndim
        ), f"Bad data ndim ({self.data.ndim} != {self.target_ndim})"
        if self.color is not None:
            self.color = x_to_ndarray(self.color).astype(np.int32)

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
        homo_data = np.hstack(
            (point.data[:, :3], np.ones((point.point_num, 1), dtype="float32"))
        )
        return cls(homo_data)

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
            _, x, y, z, r, g, b, error, *args = [float(i) for i in line.split(" ")]
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
            assert savename.endswith(".pcd"), "savename should endswith `.pcd`"
            o3d.io.write_point_cloud(savename, pcd, print_progress=True)
        else:
            return pcd

    @classmethod
    def from_open3d(cls, pcd_obj: Union[str, "PointCloud"]):
        assert OPEN3D_INSTALLED, "`to_open3d` requires `open3d` installed."
        if isinstance(pcd_obj, str) and os.path.exists(pcd_obj):
            pcd = o3d.io.read_point_cloud(pcd_obj)
        elif isinstance(pcd_obj, o3d.geometry.PointCloud):
            pcd = pcd_obj
        else:
            raise TypeError(f"Unknown type: {pcd_obj}({type(pcd_obj)})")
        color = np.asarray(pcd.colors) if pcd.has_colors() else None
        return cls(np.asarray(pcd.points), color=color)


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

    def __post_init__(self):  # type: ignore
        if len(self.data) == 4:
            fx, fy, x, y = self.data
            self.data = np.array(
                [
                    [fx, 0.0, x],
                    [0.0, fy, y],
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
        return self.cx * 2

    @property
    def h(self):
        return self.cy * 2

    def tolist(self) -> List[float]:
        return [self.fx, self.fy, self.cx, self.cy]


@dataclass
class Camera:

    extrinsic: CameraExtrinsic
    intrinsic: CameraIntrinsic

    def __init__(self, extrinsic, intrinsic):
        self.extrinsic = CameraExtrinsic(extrinsic)
        self.intrinsic = CameraIntrinsic(intrinsic)

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

        image = draw_point(image, point_image, opacity=0.5)
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
        world2cam = np.linalg.inv(extrinsic.data)
        point_camera = np.dot(homo_points.data, world2cam.T)
        point_camera /= point_camera[:, -1].reshape((-1, 1))
        return PointcloudTensor(point_camera[:, :3], color=point.color)

    @staticmethod
    def camera_to_image(
        point: PointcloudTensor,
        intrinsic: CameraIntrinsic,
        dist_coeffs: Optional[Tensor] = None,
    ) -> PointcloudTensor:
        """Project point in camera space to image space."""

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
        return PointcloudTensor(point_image, color=point.color)

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
            extrinsic = np.asarray(frame['transform_matrix'])
            camera = cls(extrinsic, intrinsic)
            cameras.append(camera)
        return cameras


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

    _KEYS = ["camera_angle_x", "camera_angle_y", "fl_x", "fl_y",
             "k1", "k2", "p1", "p2", "cx", "cy", "w", "h", "aabb_scale"]

    def __init__(self, data: Dict, sort_key="file_path"):
        self._data = data
        assert "frames" in self._data
        self._frames = self._data["frames"]
        if sort_key:
            self._frames = sorted(self._frames, key=lambda x : x["file_path"])

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        elif name in self._KEYS:
            raise AttributeError("attr name: {name} not found in json file.")

    def __len__(self):
        return len(self._frames)

    @classmethod
    def fromfile(cls, json_file: str):
        with open(json_file, 'r') as fid:
            data = json.load(fid)
        return cls(data)

    @property
    def frames(self):
        return  self._frames

    def dump(self, output: str):
        """Dump to json file."""
        with open(output, 'w') as fid:
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
            "frames": []
        }
        for p in o3d_camera.parameters:
            extrinsic = np.asarray(p['extrinsic']).reshape(4, 4).transpose()
            item = {"file_path": None, "transform_matrix": extrinsic.tolist()}
            data["frames"].append(item)
        return cls(data, sort_key=None)

    @staticmethod
    def is_valid(json_file: str) -> bool:
        """If input json_file is a valid nerf camera pose file."""
        with open(json_file, 'r') as fid:
            data = json.load(fid)
        valid = ("frames" in data and
                  isinstance(data["frames"], list) and
                  "transform_matrix" in data["frames"][0])
        return valid

    def align_first_pose_to_origin(self) -> np.ndarray:
        """Align first pose to origin point."""
        p0 = self.frames[0]["transform_matrix"]
        t_mat = np.array([
            [1, 0, 0, -p0[0][3]],
            [0, 1, 0, -p0[1][3]],
            [0, 0, 1, -p0[2][3]],
            [0, 0, 0, 1],
        ])
        for f in self.frames:
            f["transform_matrix"] = np.matmul(t_mat, f["transform_matrix"]).tolist()
        return t_mat


class Open3dCameraTrajectory:

    def __init__(self, data: Dict):
        self._data = data
        assert "parameters" in data
        self._parameters = data["parameters"]

    def dump(self, output: str):
        """Dump to json file."""
        with open(output, 'w') as fid:
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
            "intrinsic_matrix": [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]
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
        with open(json_file, 'r') as fid:
            data = json.load(fid)
        return cls(data)

    def __len__(self):
        return len(self._parameters)

    @property
    def parameters(self):
        return  self._parameters

    @staticmethod
    def is_valid(json_file: str) -> bool:
        """If input json_file is a valid open3d camera trajectory file."""
        with open(json_file, 'r') as fid:
            data = json.load(fid)
        valid = ("parameters" in data and
                  isinstance(data["parameters"], list) and
                  "extrinsic" in data["parameters"][0])
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

    def name_to_id(self, name: str) -> str:
        """Convert image name to image id."""
        if not hasattr(self, "_name_to_id"):
            self._name_to_id = {item["IMAGE_NAME"]: key
                    for key, item in self.data.items()}
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
                    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = line.strip().split(" ")
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
                    item = np.asarray([float(x) for x in line.strip().split(" ")])
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
                POINT3D_ID, X, Y, Z, R, G, B, ERROR, *TRACK = line.strip().split(" ")
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
