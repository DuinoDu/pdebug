############### utils/structs.py ###############
import abc
import json
import math
import shutil
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from pdebug.utils.env import (
    PYRENDER_INSTALLED,
    SCIPY_INSTALLED,
    TORCH_INSTALLED,
    TRIMESH_INSTALLED,
)

import numpy as np

if TORCH_INSTALLED:
    import torch
if TRIMESH_INSTALLED:
    import trimesh  # version >= 4.7.4
if PYRENDER_INSTALLED:
    import pyrender

    pyrender_RenderFlags_None = pyrender.constants.RenderFlags.NONE
else:
    pyrender_RenderFlags_None = 0

if SCIPY_INSTALLED:
    import scipy

import json
import math
import os
import shutil
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.piata.coco.utils import binary_mask_to_bbox
from pdebug.utils.ddd import get_3D_corners, load_points_3d_from_cad

import cv2
import numpy as np
import typer
import yaml

## from utils import geometry

ArrayData = Union[np.ndarray, torch.Tensor]
Color = Tuple[float, float, float]


class RigidTransform(NamedTuple):
    """An N-D rigid transformation.

    R: An NxN rotation matrix.
    t: An Nx1 translation vector
    """

    R: ArrayData
    t: ArrayData

    def inv(self):
        T_inv = np.linalg.inv(get_rigid_matrix(self))
        return RigidTransform(T_inv[:3, :3], T_inv[:3, 3:4])


ObjectPose = RigidTransform


class ObjectAnnotation(NamedTuple):
    """An annotation of a single object instance.

    dataset: The name of the dataset to which the object belogs (e.g. "tless").
    lid: A local ID (e.g. BOP IDs: https://bop.felk.cvut.cz/datasets/).
    uid: A globally unique ID (e.g. an FBID of an object model saved in Gaia).
    pose: The object pose (a transformation from the model to the world).
    masks_modal: Per-view 2D modal masks (of the same size as the image).
    masks_amodal: Per-view 2D modal masks (of the same size as the image).
    boxes_modal: Per-view 2D modal bounding boxes (of the same size as the image).
    boxes_amodal: Per-view 2D amodal bounding boxes (of the same size as the image).
    active: Binary signal whether the object is active (used for action
        recognition, e.g., for the Toy Assembly demo).
    visibilities: Per-view fractions of the object silhouette that is visible.
    """

    dataset: str
    lid: int
    uid: Optional[str] = None
    pose: Optional[ObjectPose] = None
    masks_modal: Optional[ArrayData] = None
    masks_amodal: Optional[ArrayData] = None
    boxes_modal: Optional[ArrayData] = None
    boxes_amodal: Optional[ArrayData] = None
    active: Optional[bool] = None
    visibilities: Optional[ArrayData] = None


class ImageObjectAnnotations(NamedTuple):
    """Annotations of objects in an image.

    Having annotations of all objects from an image in one structure is convenient
    for single-image methods such as Mask R-CNN.

    labels: Class labels of shape (num_objs).
    poses: 3D transformation matrices of shape (num_objs, 4, 4). The
        transformations are from the model to the camera space.
    masks: Object masks of shape (num_objs, im_height, im_width).
    boxes: 2D bounding boxes of shape (num_objs, 4).
    active_labels: Active labels of shape (num_objs).
    visibilities: Visible fractions of shape (num_objs).
    """

    labels: Optional[ArrayData] = None
    poses: Optional[ArrayData] = None
    masks: Optional[ArrayData] = None
    boxes: Optional[ArrayData] = None
    active_labels: Optional[ArrayData] = None
    visibilities: Optional[ArrayData] = None


class SceneAnnotation(NamedTuple):
    """A data sample/unit of a Torch dataset.

    image: Color/monochrome images of one of the following shapes:
        (1) (num_frames, num_views, num_channels, im_height, im_width)
        (2) (num_frames, num_views, im_height, im_width, num_channels)
    depth_image: Depth images of shape (num_frames, num_views, im_height, im_width).
    camera: Per-view camera properties (cameras[i][j] are camera properties for
        view j of frame i).
    objects_anno: Per-frame object annotations (objects_anno[i][o] are annotations
        for object o in frame i).
    """

    image: Optional[ArrayData] = None
    depth_image: Optional[ArrayData] = None
    camera: Optional[List[List["CameraModel"]]] = None
    objects_anno: Optional[List[List[ObjectAnnotation]]] = None


class AlignedBox2f:
    """
    An 2D axis aligned box in floating point.

    Assumptions:
    * The origin is at top-left corner
    * `right` and `bottom` are not inclusive in the region, i.e. the width and
        height can be simply calculated by `right - left` and `bottom - top`.
    * This is the implementation of definition 2 in this doc:
        https://docs.google.com/document/d/14YNR6ebjMgnPaeP0-RsNTEjyCHznwRLDqWmHKBNXMLo/edit
    """

    def __init__(self, left: float, top: float, right: float, bottom: float):
        """Initializes the bounding box given (left, top, right, bottom)"""
        self._left: float = left
        self._top: float = top
        self._right: float = right
        self._bottom: float = bottom

    def __repr__(self):
        return f"AlignedBox2f(left: {self._left}, top: {self._top}, right: {self._right}, bottom: {self._bottom})"

    @property
    def left(self) -> float:
        """Left of the aligned box on x-axis."""
        return self._left

    @property
    def top(self) -> float:
        """Top of the aligned box on y-axis."""
        return self._top

    @property
    def right(self) -> float:
        """Right of the aligned box on x-axis."""
        return self._right

    @property
    def bottom(self) -> float:
        """Bottom of the aligned box on y-axis."""
        return self._bottom

    @property
    def width(self) -> float:
        """Width of the aligned box.

        Returns:
            Width computed by right - left
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height of the aligned box.

        Returns:
            Height computed by bottom - top
        """
        return self.bottom - self.top

    def pad(self, width: float, height: float) -> "AlignedBox2f":
        """Pads the region by extending `width` and `height` on four sides.

        Args:
            width (float): length to pad on left and right sides
            height (float): length to pad on top and bottom sides
        Returns:
            a new AlignedBox2f object with padded region
        """
        return AlignedBox2f(
            self.left - width,
            self.top - height,
            self.right + width,
            self.bottom + height,
        )

    def array_ltrb(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,):  (left, top, right, bottom).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return np.array([self.left, self.top, self.right, self.bottom])

    def array_ltwh(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return np.array([self.left, self.top, self.width, self.height])

    def int_array_ltrb(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return self.array_ltrb().astype(int)

    def int_array_ltwh(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return self.array_ltwh().astype(int)

    def round(self) -> "AlignedBox2f":
        """Rounds the float values to int.

        Returns:
            a new AlignedBox2f object with rounded values (still float)
        """
        return AlignedBox2f(
            np.round(self.left),
            np.round(self.top),
            np.round(self.right),
            np.round(self.bottom),
        )

    def clip(self, boundary: "AlignedBox2f") -> "AlignedBox2f":
        """Clips the region by the boundary

        Args:
            boundary (AlignedBox2f): boundary of box to be clipped
                (boundary.left: minimum left / right value,
                 boundary.top: minimum top / bottom value,
                 boundary.right: maximum left / right value,
                 boundary.bottom: maximum top / bottom value)
        Returns:
            a new clipped AlignedBox2f object
        """
        return AlignedBox2f(
            min(max(self.left, boundary.left), boundary.right),
            min(max(self.top, boundary.top), boundary.bottom),
            min(max(self.right, boundary.left), boundary.right),
            min(max(self.bottom, boundary.top), boundary.bottom),
        )


class CameraModel(abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    T_world_from_eye : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's extrinsics after construction by
        assigning to or modifying this matrix.

    serial : string
        Arbitrary string identifying the specific camera.

    Attributes
    ----------
    Most attributes are the same as constructor parameters.

    zmin
        Smallest z coordinate of a visible unit-length eye vector.
        (Unit-length) eye rays with z < zmin are known not to be visible
        without doing any extra work.

        This check is needed because for points far outside the window,
        as the distortion polynomial explodes and incorrectly maps some
        distant points back to coordinates inside the window.

        `zmin = cos(max_angle)`

    max_angle
        Maximum angle from +Z axis of a visible eye vector.
    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    T_world_from_eye: np.ndarray

    _zmin: Optional[float]
    _max_angle: Optional[float]

    def __init__(
        self,
        width,
        height,
        f,
        c,
        T_world_from_eye=None,
        serial="",
    ):  # pylint: disable=super-init-not-called (see issue 4790 on pylint github)
        self.width = width
        self.height = height
        self.serial = serial

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if T_world_from_eye is None:
            self.T_world_from_eye = np.eye(4)
        else:
            self.T_world_from_eye = as_4x4(T_world_from_eye, copy=True)
            if (
                np.abs(
                    (self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3]
                    - np.eye(3)
                ).max()
                >= 1.0e-5
            ):
                info_str = (
                    "camera T_world_from_eye must be a rigid transform\n"
                )
                info_str = info_str + "T\n{}\n".format(self.T_world_from_eye.T)
                info_str = info_str + "(T*T_t - I).max()\n{}\n".format(
                    np.abs(
                        (self.T_world_from_eye.T @ self.T_world_from_eye)[
                            :3, :3
                        ]
                        - np.eye(3)
                    ).max()
                )
                raise ValueError(info_str)

        # These are computed only when needed, use the getters zmin() and max_angle()
        self._zmin = None
        self._max_angle = None

    def __repr__(self):
        return f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"

    def to_json(self):
        js = {}
        js["ImageSizeX"] = self.width
        js["ImageSizeY"] = self.height
        js["T_WorldFromCamera"] = self.T_world_from_eye.tolist()

        js["ModelViewMatrix"] = np.linalg.inv(self.T_world_from_eye).tolist()

        js["fx"], js["fy"] = np.asarray(self.f).tolist()
        js["cx"], js["cy"] = np.asarray(self.c).tolist()

        return js

    def copy(
        self,
        T_world_from_eye=None,
    ):
        """Return a copy of this camera

        Arguments
        ---------
        T_world_from_eye : 4x4 np.ndarray
            Optional new extrinsics for the new camera model.
            Default is to copy this camera's extrinsics.

        serial : str
            Optional replacement serial number.
            Default is to copy this camera's serial number.
        """
        return self.crop(
            0,
            0,
            self.width,
            self.height,
            T_world_from_eye=T_world_from_eye,
        )

    def compute_zmin(self):
        corners = (
            np.array(
                [
                    [0, 0],
                    [self.width, 0],
                    [0, self.height],
                    [self.width, self.height],
                ]
            )
            - 0.5
        )
        self._zmin = self.window_to_eye(corners)[:, 2].min()
        self._max_angle = np.arccos(self._zmin)

    def zmin(self):
        if self._zmin is None:
            self.compute_zmin()
        return self._zmin

    def max_angle(self):
        if self._max_angle is None:
            self.compute_zmin()
        return self._max_angle

    def world_to_window(self, v):
        """Project world space points to 2D window coordinates"""
        return self.eye_to_window(self.world_to_eye(v))

    def world_to_window3(self, v):
        """Project world space points to 3D window coordinates (uv + depth)"""
        return self.eye_to_window3(self.world_to_eye(v))

    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):
        # map to [u,v,1] and renormalize
        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = normalized(v, axis=-1)
        return v

    @staticmethod
    def project3(v):
        # map to [x/z, y/z, z]
        x, y, z = np.moveaxis(v, -1, 0)
        return np.stack([x / z, y / z, z], axis=-1)

    @staticmethod
    def unproject3(p):
        # map to [p*z, v*z, z]
        x, y, z = np.moveaxis(p, -1, 0)
        return np.stack((x * z, y * z, z), axis=-1)

    def pos(self):
        """Return world position of camera"""
        return self.T_world_from_eye[:3, 3]

    def orient(self):
        """Return world orientation of camera as 3x3 matrix"""
        return self.T_world_from_eye[:3, :3]

    def window_to_world_ray(self, w):
        """
        Unproject 2D window coordinates to world rays.

        Returns a tuple of (origin, direction)
        """
        v = rotate_points(self.T_world_from_eye, self.window_to_eye(w))
        o = np.broadcast_to(self.pos(), v.shape)
        return (o, v)

    def window_to_world3(self, w):
        """Unproject 3D window coordinates (uv + depth) to world points"""
        return self.eye_to_world(self.window_to_eye3(w))

    def world_visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def world_to_eye(self, v):
        """
        Apply camera inverse extrinsics to points `v` to get eye coords
        """
        return rotate_points(
            self.T_world_from_eye.T, v - self.T_world_from_eye[:3, 3]
        )

    def eye_to_world(self, v):
        """
        Apply camera extrinsics to eye points `v` to get world coords
        """
        return transform_points(self.T_world_from_eye, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        return p * self.f + self.c

    def window_to_eye(self, w):
        """Unproject 2d window coordinates to unit-length 3D eye coordinates"""

        q = (np.asarray(w) - self.c) / self.f
        return self.unproject(q)

    def eye_to_window3(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p

    def window_to_eye3(self, w):
        """Unproject 3d window coordinates (uv + depth) to eye coordinates"""
        assert self.undistort is not None
        temp = np.array(w, dtype=np.float64)
        temp[..., :2] -= self.c
        temp[..., :2] /= self.f
        temp[..., :2] = self.undistort.evaluate(temp[..., :2])
        return self.unproject3(temp)

    def visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def eye_visible(self, v):
        """
        Returns true if the given eye points are visible in this camera
        """
        v = normalized(v)
        w = self.eye_to_window(v)
        return (v[..., 2] >= self.zmin()) & self.w_visible(w)

    def w_visible(self, w, *, margin=0):
        """
        Return True if the 2d window coordinate `w` is inside the window

        Can be called with an array, returning a bool array.
        """
        x, y = np.moveaxis(w, -1, 0)
        x0 = -margin - 0.5
        y0 = -margin - 0.5
        x1 = self.width + margin - 0.5
        y1 = self.height + margin - 0.5
        return (x > x0) & (x < x1) & (y >= y0) & (y < y1)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        T_world_from_eye=None,
        serial=None,
    ):
        """
        Return intrinsics for a crop of the sensor image.

        No scaling is applied; this just returns the model for a sub-
        array of image data. (Or for a larger array, if (x,y)<=0 and
        (width, height) > (self.width, self.height).

        To do both cropping and scaling, use :meth:`subrect`

        Parameters
        ----------
        x, y, width, height
            Location and size in this camera's window coordinates
        """
        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.T_world_from_eye
            if T_world_from_eye is None
            else T_world_from_eye,
            self.serial if serial is None else serial,
        )

    def subrect(
        self, transform, width, height, bypass_fit_undistort_coeffs=False
    ):
        """
        Return intrinsics for a scaled crop of the sensor image.

        Parameters
        ----------
        Transform
            a 2x3 affine transform matrix that takes coordinates in the
            old image rect to coordinates in the new image rect, as for
            `cv.WarpAffine`.

            The transform is given in continuous coords, so it must
            follow the "pixel center on integer grid coordinates"
            convention. E.g. resizing an image by 1/N is not just
            scaling by 1/N, but scaling by `1/N` and translating by
            `(1-N)/(2N)`

            Yes, this is confusing. Blame the CV community for failing
            to learn anything from the graphics community.

        width, height : int
            size of target image
        """
        # Currently only support scale and translation.
        #
        # We could support 90 degree rotation by careful manipulation of polynomial
        # coefficients, or arbitrary rotation by applying an 2D affine transform
        # instead of just (f, c) to convert distorted coords to window coords.
        f = np.diag(transform[:-2, :-2])
        c = transform[:-2, 2]
        offdiag = np.diag(np.flip(transform[..., :2, :2], -1))
        if not np.all(offdiag == 0.0):
            raise NotImplementedError(
                "transforms with rotation not yet supported"
            )
        cam = type(self)(
            width,
            height,
            self.f * f,
            self.c * f + c,
            self.distort_coeffs,
            self.undistort_coeffs,
            self.T_world_from_eye,
            self.serial,
            bypass_fit_undistort_coeffs,
        )
        return cam

    def to_model(self, cls):
        """Convert to a different distortion model.

        Since this relies on the fitted coefficients of the first model,
        it cannot be as accurate as directly fitting the desired model from
        measurements, but it's better than nothing.
        """
        w, h = self.width, self.height
        rays, _ = _gen_rays_in_window(self, spacing=0.05)
        # forward project again, because the unproject coefficients may
        # have been fitted from the forward projection and so have extra
        # error.
        w_pts = self.eye_to_window(rays)
        return cls.fit_from_points(
            rays, w_pts, w, h, self.T_world_from_eye, self.serial
        )

    @classmethod
    def fit_from_points(
        cls, eye_pts, w_pts, width, height, T_world_from_eye=None, serial=""
    ):
        """Fit intrinsics from points with corresponding eye vectors"""
        p_pts = cls.project(eye_pts)

        # to make the solve faster and more stable:
        # first solve for just approximate (f, cx, cy), ignoring distortion
        fc_x0 = (width**2 + height**2) ** 0.5 / 4, width / 2, height / 2
        f, cx, cy = dis.fit_coeffs(
            lambda coeffs, p: p * coeffs[0] + coeffs[1:],
            p_pts,
            w_pts,
            x0=fc_x0,
        )

        # ...then solve for full distortion coefficients
        coeffs = dis.fit_coeffs(
            dis.add_f_c_coeffs(cls.distortion_model.evaluate),
            p_pts,
            w_pts,
            x0=(0,) * len(cls.distortion_model._fields) + (f, cx, cy),
        )
        f, cx, cy = coeffs[-3:]
        coeffs = coeffs[:-3]

        q_pts = (w_pts - [cx, cy]) / f
        uncoeffs = dis.fit_coeffs(cls.distortion_model, q_pts, p_pts)

        return cls(
            width,
            height,
            f,
            (cx, cy),
            coeffs,
            uncoeffs,
            T_world_from_eye,
            serial,
        )


class PinholePlaneCameraModel(CameraModel):

    model_fov_limit = 50 * (math.pi / 180)

    def uv_to_window_matrix(self):
        """Return the 3x3 intrinsics matrix"""
        return np.array(
            [[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]]
        )


############### utils/geometry.py ###############

#!/usr/bin/env python3

import math
from typing import Tuple, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

## from utils import geometry


AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")


def transform_3d_points_numpy(
    trans: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Transform 3D points. Compute trans * points

    Args:
        points: 3D points of shape (num_points, 3).
        trans: Transformation matrix of shape (4, 4).
    Returns:
        Transformed 3D points of shape (num_points, 3).
    """

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return trans.dot(points_h.T)[:3, :].T


def transform_3d_points_torch(
    trans: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Transforms sets of 3D points.

    Args:
        points: 3D points of shape (num_points, 3).
        trans: Transformation matrix of shape (4, 4).
    Returns:
        Transformed points of shape (batch_size, num_points, 3).
    """

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    assert trans.device == points.device
    assert trans.dtype == points.dtype
    points_h = torch.hstack(
        [points, torch.ones((points.shape[0], 1), device=points.device)]
    )
    return torch.matmul(trans, points_h.T)[:3, :].T


def gen_look_at_matrix(
    orig_camera_from_world: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
    return_camera_from_world: bool = True,
) -> np.ndarray:
    """
    Rotates the input camera such that the new transformation align the z-direction to the provided point in world.
    Args:
      camera_angle is used to apply a roll rotation around the new z
      return_camera_from_world is used to return the inverse

    Returns:
        world_from_aligned_camera or aligned_camera_from_world
    """

    center_local = transform_points(orig_camera_from_world, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_world_from_camera = np.linalg.inv(orig_camera_from_world)

    world_from_aligned_camera = orig_world_from_camera.copy()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ delta_r_local
    )

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler(
        "z", camera_angle, degrees=True
    ).as_matrix()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ z_local_rot
    )

    if return_camera_from_world:
        return np.linalg.inv(world_from_aligned_camera)
    return world_from_aligned_camera


def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Transform an array of 3D points with an SE3 transform (rotation and translation).

    *WARNING* this function does not support arbitrary affine transforms that also scale
    the coordinates (i.e., if a 4x4 matrix is provided as input, the last row of the
    matrix must be `[0, 0, 0, 1]`).

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points [..., 3]

    Returns:
        Transformed points [..., 3]
    """
    return rotate_points(matrix, points) + matrix[..., :3, 3]


def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Rotates an array of 3D points with an affine transform,
    which is equivalent to transforming an array of 3D rays.

    *WARNING* This ignores the translation in `m`; to transform 3D *points*, use
    `transform_points()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points or 3d direction vectors [..., 3]

    Returns:
        Rotated points / direction vectors [..., 3]
    """
    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)


def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    # Convert the vectors to unit vectors.
    a = normalized(a_orig)
    b = normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = (
        np.eye(3, 3, dtype=a_orig.dtype)
        + v_mat
        + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))
    )

    return rot


def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res


def rotation_matrix_numpy(angle: float, direction: np.ndarray) -> np.ndarray:
    """Return a homogeneous transformation matrix [4x4] to rotate a point around the
    provided direction by a mangnitude set by angle.

    Args:
        angle: Angle to rotate around axis [rad].
        direction: Direction vector (3-vector, does not need to be normalized)

    Returns:
        M: A 4x4 matrix with the rotation component set and translation to zero.

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = normalized(direction[:3])
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)),
        dtype=np.float64,
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    return M


def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype),
                    a.shape[:-2] + (1, 4),
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def normalized(
    v: AnyTensor, axis: int = -1, eps: float = 5.43e-20
) -> AnyTensor:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d


############### utils/logging.py ###############
"""
Utilities for more useful logging output.
"""

import logging
import traceback
from typing import Union

import numpy as np

FORMAT_PREFIX = (
    "%(levelname).1s%(asctime)s.%(msecs)d %(process)d %(filename)s:%(lineno)d]"
)
FORMAT = f"{FORMAT_PREFIX} %(message)s"
DATEFMT = "%m%d %H:%M:%S"


# Helper for things that format as brackets around comma-separated items
def _seq_repr(parts, prefix, suffix, indent):
    if len(parts) <= 1 or sum(len(p) for p in parts) < 80:
        return prefix + ", ".join(parts) + suffix
    return (
        f"{prefix}\n{indent}  "
        + f",\n{indent}  ".join(parts)
        + f"\n{indent}{suffix}"
    )


class LocalsFormatter(logging.Formatter):
    """
    logging.Formatter which shows local variables in stack dumps.
    """

    def formatException(self, exc_info) -> str:
        tb = traceback.TracebackException(*exc_info, capture_locals=True)
        # try not to be too verbose
        for frame in tb.stack:
            if len(repr(frame.locals)) > 500:
                frame.locals = None
        return "".join(tb.format())


def config_logging(
    *,
    fmt: str = FORMAT,
    level: Union[int, str] = logging.INFO,
    datefmt: str = DATEFMT,
    style: str = "%",
    stream=None,
) -> None:
    """
    Configure logging.

    Same as `logging.basicConfig(fmt, level, datefmt, style, force=True)`,
    except it uses :class:`LocalsFormatter` to show local variables in
    exception stack traces.

    stream: If specified, the root logger will use it for logging output; otherwise,
        sys.stderr will be used.
    """
    root = logging.getLogger()

    # always do 'force'
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    h = logging.StreamHandler(stream)
    h.setFormatter(LocalsFormatter(fmt, datefmt, style))
    root.addHandler(h)
    root.setLevel(level)


DEBUG: int = logging.DEBUG
INFO: int = logging.INFO
WARNING: int = logging.WARNING
ERROR: int = logging.ERROR

WHITE = "\x1b[37;20m"
WHITE_BOLD = "\x1b[37;1m"
BLUE = "\x1b[34;20m"
BLUE_BOLD = "\x1b[34;1m"
RESET = "\x1b[0m"

Logger = logging.Logger


def get_logger(level: int = logging.INFO) -> Logger:
    """Provides a logger with the specified logging level.

    Returns:
        A logger.
    """

    config_logging(level=level)
    return logging.getLogger(__name__)


def get_separator(length: int = 80) -> str:
    """Return a text separator to be used in logs.

    Args:
        length: Length of the separator (in the number of characters).
    """

    return length * "-"


def log_heading(logger: Logger, msg: str, style: str = WHITE) -> None:
    """Logs a visually distinct heading.

    Args:
        logger: A logger.
        heading: The heading to print.
    """

    separator = get_separator()
    logger.info(style + separator + RESET)
    logger.info(style + msg + RESET)
    logger.info(style + separator + RESET)


############### utils/misc.py ###############

#!/usr/bin/env python3

"""Miscellaneous functions."""

import dataclasses
import math
import time
import uuid
from collections import namedtuple
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import cv2
import numpy as np
from PIL import Image

## from utils import geometry, logging

## import utils.structs as structs
## from utils.structs import AlignedBox2f, CameraModel, PinholePlaneCameraModel

## from utils.geometry import transform_3d_points_numpy, gen_look_at_matrix

logger: logging.Logger = get_logger()


class Timer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_time = None

    def start(self):
        if self.enabled:
            self.start_time = time.time()

    def elapsed(self, msg="Elapsed") -> Optional[float]:
        if self.enabled:
            elapsed = time.time() - self.start_time
            logger.info(f"{msg}: {elapsed:.5f}s")
            return elapsed
        else:
            return None


def fibonacci_sampling(
    n_pts: int, radius: float = 1.0
) -> List[Tuple[float, float, float]]:
    """Fibonacci-based sampling of points on a sphere.

    Samples an odd number of almost equidistant 3D points from the Fibonacci
    lattice on a unit sphere.

    Ref:
    [1] https://arxiv.org/pdf/0912.4540.pdf
    [2] http://stackoverflow.com/questions/34302938/map-point-to-closest-point-on-fibonacci-lattice
    [3] http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    [4] https://www.openprocessing.org/sketch/41142

    Args:
        n_pts: Number of 3D points to sample (an odd number).
        radius: Radius of the sphere.
    Returns:
        List of 3D points on the sphere surface.
    """

    # Needs to be an odd number [1].
    assert n_pts % 2 == 1

    n_pts_half = int(n_pts / 2)

    phi = (math.sqrt(5.0) + 1.0) / 2.0  # Golden ratio.
    phi_inv = phi - 1.0
    ga = 2.0 * math.pi * phi_inv  # Complement to the golden angle.

    pts = []
    for i in range(-n_pts_half, n_pts_half + 1):
        lat = math.asin((2 * i) / float(2 * n_pts_half + 1))
        lon = (ga * i) % (2 * math.pi)

        # Convert the latitude and longitude angles to 3D coordinates.
        # Latitude (elevation) represents the rotation angle around the X axis.
        # Longitude (azimuth) represents the rotation angle around the Z axis.
        s = math.cos(lat) * radius
        x, y, z = math.cos(lon) * s, math.sin(lon) * s, math.tan(lat) * s
        pts.append([x, y, z])

    return pts


def sample_views(
    min_n_views: int,
    radius: float = 1.0,
    azimuth_range: Tuple[float, float] = (0, 2 * math.pi),
    elev_range: Tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi),
    mode: str = "fibonacci",
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Viewpoint sampling from a view sphere.

    Args:
        min_n_views: The min. number of points to sample on the whole sphere.
        radius: Radius of the sphere.
        azimuth_range: Azimuth range from which the viewpoints are sampled.
        elev_range: Elevation range from which the viewpoints are sampled.
        mode: Type of sampling (options: "fibonacci").
    Returns:
        List of views, each represented by a 3x3 ndarray with a rotation
        matrix and a 3x1 ndarray with a translation vector.
    """

    # Get points on a sphere.
    if mode == "fibonacci":
        n_views = min_n_views
        if n_views % 2 != 1:
            n_views += 1

        pts = fibonacci_sampling(n_views, radius=radius)
        pts_level = [0 for _ in range(len(pts))]
    else:
        raise ValueError("Unknown view sampling mode.")

    views = []
    for pt in pts:
        # Azimuth from (0, 2 * pi).
        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        # Elevation from (-0.5 * pi, 0.5 * pi).
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        if not (
            azimuth_range[0] <= azimuth <= azimuth_range[1]
            and elev_range[0] <= elev <= elev_range[1]
        ):
            continue

        # Rotation matrix.
        # Adopted from gluLookAt function (uses OpenGL coordinate system):
        # [1] http://stackoverflow.com/questions/5717654/glulookat-explanation
        # [2] https://www.opengl.org/wiki/GluLookAt_code
        f = -np.array(pt)  # Forward direction.
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0])  # Up direction.
        s = np.cross(f, u)  # Side direction.
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis.
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)  # Recompute up.
        R = np.array(
            [[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]]
        )

        # Convert from OpenGL to OpenCV coordinate system.
        R_yz_flip = rotation_matrix_numpy(math.pi, np.array([1, 0, 0]))[:3, :3]
        R = R_yz_flip.dot(R)

        # Translation vector.
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({"R": R, "t": t})

    return views, pts_level


def calc_crop_box(
    box: AlignedBox2f,
    box_scaling_factor: float = 1.0,
    make_square: bool = False,
) -> AlignedBox2f:
    """Adjusts a bounding box to the specified aspect and scale.

    Args:
        box: Bounding box.
        box_aspect: The aspect ratio of the target box.
        box_scaling_factor: The scaling factor to apply to the box.
    Returns:
        Adjusted box.
    """

    # Potentially inflate the box and adjust aspect ratio.
    crop_box_width = box.width * box_scaling_factor
    crop_box_height = box.height * box_scaling_factor

    # Optionally make the box square.
    if make_square:
        crop_box_side = max(crop_box_width, crop_box_height)
        crop_box_width = crop_box_side
        crop_box_height = crop_box_side

    # Calculate padding.
    x_pad = 0.5 * (crop_box_width - box.width)
    y_pad = 0.5 * (crop_box_height - box.height)

    return AlignedBox2f(
        left=box.left - x_pad,
        top=box.top - y_pad,
        right=box.right + x_pad,
        bottom=box.bottom + y_pad,
    )


def construct_crop_camera(
    box: AlignedBox2f,
    camera_model_c2w: CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float,
) -> CameraModel:
    """Constructs a virtual pinhole camera from the specified 2D bounding box.

    Args:
        camera_model_c2w: Original camera model with extrinsics set to the
            camera->world transformation.

        viewport_crop_size: Viewport size of the new camera.
        viewport_scaling_factor: Requested scaling of the viewport.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center
        of the specified 2D bounding box and whose focal length is set such as
        the sphere representing the bounding box (+ requested padding) is visible
        in the camera viewport.
    """

    # Get centroid and radius of the reference sphere (the virtual camera will
    # be constructed such as the projection of the sphere fits the viewport.
    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    box_corners_in_c = np.array(
        [
            [box.left - cx, box.top - cy, f],
            [box.right - cx, box.top - cy, f],
            [box.left - cx, box.bottom - cy, f],
            [box.right - cx, box.bottom - cy, f],
        ]
    )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[
        :3, 0
    ]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    # Transformations from world to the original and virtual cameras.
    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = gen_look_at_matrix(trans_w2c, centroid_in_w)

    # Transform the centroid from world to the virtual camera.
    centroid_in_vc = transform_3d_points_numpy(
        trans_w2vc, np.expand_dims(centroid_in_w, axis=0)
    ).squeeze()

    # Project the sphere radius to the image plane of the virtual camera and
    # enlarge it by the specified padding. This defines the 2D extent that
    # should be visible in the virtual camera.
    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = fx_fy_orig * cx_cy / extent_2d

    # Parameters of the virtual camera.
    return PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )


def calc_2d_box(
    xs: torch.Tensor,
    ys: torch.Tensor,
    im_size: Optional[torch.Tensor] = None,
    clip: bool = False,
) -> torch.Tensor:
    """Calculates the 2D bounding box of a set of 2D points.

    Args:
        xs: A 1D tensor with x-coordinates of 2D points.
        ys: A 1D tensor with y-coordinates of 2D points.
        im_size: The image size (width, height), used for optional clipping.
        clip: Whether to clip the bounding box (default == False).
    Returns:
        The 2D bounding box (x1, y1, x2, y2), where (x1, y1) and (x2, y2) is the
        minimum and the maximum corner respectively.
    """
    if len(xs) == 0 or len(ys) == 0:
        return torch.Tensor([0.0, 0.0, 0.0, 0.0])

    box_min = torch.as_tensor([xs.min(), ys.min()])
    box_max = torch.as_tensor([xs.max(), ys.max()])
    if clip:
        if im_size is None:
            raise ValueError("Image size needs to be provided for clipping.")
        box_min = clip_2d_point(box_min, im_size)
        box_max = clip_2d_point(box_max, im_size)
    return torch.hstack([box_min, box_max])


def get_rigid_matrix(trans: RigidTransform) -> np.ndarray:
    """Creates a 4x4 transformation matrix from a 3x3 rotation and 3x1 translation.

    Args:
        trans: A rigid transformation defined by a 3x3 rotation matrix and
            a 3x1 translation vector.
    Returns:
        A 4x4 rigid transformation matrix.
    """

    matrix = np.eye(4)
    matrix[:3, :3] = trans.R
    matrix[:3, 3:] = trans.t
    return matrix


def get_intrinsic_matrix(cam: CameraModel) -> np.ndarray:
    """Returns a 3x3 intrinsic matrix of the given camera.

    Args:
        cam: The input camera model.
    Returns:
        A 3x3 intrinsic matrix K.
    """

    return np.array(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:
    """Resizes an image.

    Args:
      image: An input image.
      size: The size of the output image (width, height).
      interpolation: An interpolation method (a suitable one is picked if undefined).
    Returns:
      The resized image.
    """

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
        )
    return cv2.resize(image, size, interpolation=interpolation)


def map_fields(func, obj, only_type=object):
    """
    map 'func' recursively over nested collection types.

    >>> map_fields(lambda x: x * 2,
    ...            {'a': 1, 'b': {'x': 2, 'y': 3}})
    {'a': 2, 'b': {'x': 4, 'y': 6}}

    E.g. to detach all tensors in a network output frame:

        frame = map_fields(torch.detach, frame, torch.Tensor)

    The optional 'only_type' parameter only calls `func` for values where
    isinstance(value, only_type) returns True. Other values are returned
    as-is.
    """
    if is_dictlike(obj):
        ty = type(obj)
        if isinstance(obj, Mapping):
            return ty(
                (k, map_fields(func, v, only_type)) for (k, v) in obj.items()
            )
        else:
            # NamedTuple or dataclass
            return ty(
                **{
                    k: map_fields(func, v, only_type)
                    for (k, v) in asdict(obj).items()
                }
            )
    elif isinstance(obj, tuple):
        return tuple(map_fields(func, v, only_type) for v in obj)
    elif isinstance(obj, list):
        return [map_fields(func, v, only_type) for v in obj]
    elif isinstance(obj, only_type):
        return func(obj)
    else:
        return obj


def is_dictlike(obj: Any) -> bool:
    """
    Returns true if the object is a dataclass, NamedTuple, or Mapping.
    """
    return (
        dataclasses.is_dataclass(obj)
        or hasattr(obj, "_asdict")
        or isinstance(obj, Mapping)
    )


def chw_to_hwc(data: np.ndarray) -> np.ndarray:
    """Converts a Numpy array from CHW to HWC (C = channels, H = height, W = width).

    Args:
        data: A Numpy array width dimensions in the CHW order.
    Returns:
        A Numpy array width dimensions in the HWC order.
    """

    return np.transpose(data, (1, 2, 0))


def slugify(string: str) -> str:
    """Slugify a string (typically a path) such as it can be used as a filename.

    Args:
        string: A string to slugify.
    Returns:
        A slugified string.
    """
    return (
        string.strip("/").replace("/", "-").replace(" ", "-").replace(".", "-")
    )


def crop_image(image: np.ndarray, crop_box: AlignedBox2f) -> np.ndarray:
    """Crops an image.

    Args:
        image: The input HWC image.
        crop_box: The bounding box for cropping given by (x1, y1, x2, y2).
    Returns:
        Cropped image.
    """

    return image[
        crop_box.top : crop_box.bottom, crop_box.left : crop_box.right
    ]


def ensure_three_channels(im: np.ndarray) -> np.ndarray:
    """Ensures that the image has 3 channels.

    Args:
        im: The input image.
    Returns:
        An image with 3 channels (single-channel images are duplicated).
    """

    if im.ndim == 3:
        return im
    elif im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        return np.dstack([im, im, im])
    else:
        raise ValueError("Unknown image format.")


def warp_image(
    src_camera: CameraModel,
    dst_camera: CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
    factor_to_downsample: int = 1,
) -> np.ndarray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    factor_to_downsample :
        If this value is greater than 1, it will downsample the input image prior to warping.
        This improves downsampling performance, in an attempt to replicate
        area interpolation for crop+undistortion warps.
    """

    if factor_to_downsample > 1:
        src_image = cv2.resize(
            src_image,
            (
                int(src_image.shape[1] / factor_to_downsample),
                int(src_image.shape[0] / factor_to_downsample),
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Rescale source camera
        src_camera = adjust_camera_model(src_camera, factor_to_downsample)

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def warp_depth_image(
    src_camera: CameraModel,
    dst_camera: CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:

    # Copy the source depth image.
    depth_image = np.array(src_depth_image)

    # If the camera extrinsics changed, update the depth values.
    if not np.allclose(
        src_camera.T_world_from_eye, dst_camera.T_world_from_eye
    ):

        # Image coordinates with valid depth values.
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        # Transform the source depth image to a point cloud.
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(
            depth_image[valid_mask] / pts_in_src[:, 2], axis=1
        )

        # Transform the point cloud from the source to the target camera.
        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    # Warp the depth image to the target camera.
    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )


def array_to_tensor(
    array: np.ndarray, make_array_writeable: bool = True
) -> torch.Tensor:
    """Converts a Numpy array into a tensor.

    Args:
        array: A Numpy array.
        make_array_writeable: Whether to force the array to be writable.
    Returns:
        A tensor.
    """

    # If the array is not writable, make it writable or copy the array.
    # Otherwise, torch.from_numpy() would yield a warning that tensors do not
    # support the writing lock and one could modify the underlying data via them.
    if not array.flags.writeable:
        if make_array_writeable and array.flags.owndata:
            array.setflags(write=True)
        else:
            array = np.array(array)
    return torch.from_numpy(array)


def arrays_to_tensors(data: Any) -> Any:
    """Recursively converts Numpy arrays into tensors.

    Args:
        data: A possibly nested structure with Numpy arrays.
    Returns:
        The same structure but with Numpy arrays converted to tensors.
    """

    return map_fields(lambda x: array_to_tensor(x), data, only_type=np.ndarray)


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor into a Numpy array.

    Args:
        tensor: A tensor (may be in the GPU memory).
    Returns:
        A Numpy array.
    """

    return tensor.detach().cpu().numpy()


def tensors_to_arrays(data: Any) -> Any:
    """Recursively converts tensors into Numpy arrays.

    Args:
        data: A possibly nested structure with tensors.
    Returns:
        The same structure but with tensors converted to Numpy arrays.
    """

    return map_fields(
        lambda x: tensor_to_array(x), data, only_type=torch.Tensor
    )


############### utils/renderer_base.py ###############

#!/usr/bin/env python3


"""The base class for renderers."""


from enum import Enum
from typing import Any, Dict, Optional, Sequence

import numpy as np
import trimesh

## from utils import structs


def get_single_model_color(mesh: trimesh.Trimesh) -> Color:
    """Gets a single color for a mesh.

    Args:
        mesh: A mesh for which to get the color.
    Returns:
        A single color for the mesh -- either the average vertex color (if vertex
        colors are defined) or a default color.
    """

    try:
        return tuple(np.mean(mesh.visual.vertex_colors[:, :3], axis=0) / 255.0)
    except AttributeError:
        return (0.0, 1.0, 0.0)


class RenderType(Enum):
    """The rendering type.

    COLOR: An RGB image.
    DEPTH: A depth image with the depth values in mm.
    NORMAL: A normal map with normals expressed in the camera space.
    MASK: A binary mask.
    """

    COLOR = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    MASK = "mask"


class RendererBase:
    """The base class which all renderers should inherit."""

    def __init__(self, **kwargs: Any) -> None:

        raise NotImplementedError()

    def add_object_model(
        self,
        object_id: int,
        mesh_color: Optional[Color] = None,
        **kwargs: Any,
    ) -> None:
        """Adds an object model to the renderer.

        Args:
            asset_key: The key of an asset to add to the renderer.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        raise NotImplementedError()

    def render_object_model(
        self,
        object_id: int,
        camera_model_c2m: CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, ArrayData]:
        """Renders an object model in the specified pose.

        Args:
            asset_key: The key of an asset to render.
            camera_model_c2m: A camera model with the extrinsics set to a rigid
                transformation from the camera to the model frame.
            render_types: Types of images to render.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        raise NotImplementedError()

    def render_meshes(
        self,
        meshes_in_w: Sequence[trimesh.Trimesh],
        camera_model_c2w: CameraModel,
        render_types: Sequence[RenderType],
        mesh_colors: Optional[Sequence[Color]] = None,
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, ArrayData]:
        """Renders a list of meshes.

        Args:
            meshes_in_w: A list of meshes to render. The meshes are assumed to be
                defined in the same world coordinate frame.
            camera_model_c2w: A camera model with the extrinsics set to a rigid
                transformation from the camera to the world frame.
            render_types: Types of images to render.
            mesh_colors: A single color per mesh. Original mesh colors are used
                if not specified.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        raise NotImplementedError()


############### utils/json_util.py ###############
"""
Helpers for working with JSON.
"""

import collections.abc as abc
import dataclasses
import inspect
import json
import re
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

import numpy as np

# type of json objects.
#
# The last two union elements should be `List["JsonValue"]` and
# `Dict[str,"JsonValue"]`, but none of the Python type checkers
# are able to handle recursive  types.

JsonValue = Union[None, str, bool, int, float, List[Any], Dict[str, Any]]

T = TypeVar("T")


# Return the json value resulting from merging b into a.
def merge(a: JsonValue, b: JsonValue) -> JsonValue:
    # dicts merge any common elements
    if isinstance(a, dict) and isinstance(b, dict):
        a = dict(a)
        for k in b:
            if k in a:
                a[k] = merge(a[k], b[k])
            else:
                a[k] = b[k]
        return a

    # anything else (including lists) just replaces the old value
    return b


def merge_at(obj: JsonValue, path: str, rhs: JsonValue) -> JsonValue:
    """
    Return the result of merging `rhs` into `obj` at path `path`. Obj
    itself is not modified.

    Arguments:
    obj -- a json object
    path -- dotted field specifier, e.g. 'a.b.c'; empty to replace whole obj
    rhs -- new json value to merge at path

    >>> merge_at({'x': 1, 'y': {'a': 2, 'b': 3}}, 'x.b', 4)
    {'x': 1, 'y': {'a': 2, 'b': 4}}

    >>> merge_at({'x': 1, 'y': 2}, '', {'z': 3})
    {'x': 1, 'y': 2, 'z': 3}
    """
    keys = [k for k in path.split(".") if k]
    while keys:
        key = keys.pop()
        rhs = {key: rhs}
    return merge(obj, rhs)


# Parse s as a json value, but allow non-json strings to be given
# without quotes for convenience on command line.
def parse_json_or_str(s: str) -> JsonValue:
    s = s.strip()
    if re.match("""null$|true$|false$|^[0-9."'[{-]""", s):
        return json.loads(s)
    else:
        return s


# merge an assignment like 'a.b.c={some json}' into a json object
# if the rhs looks
def merge_from_arg(obj: JsonValue, arg: str):
    # look for '=' before any quotes
    if re.match("[^\"']+=", arg):
        path, rhs = arg.split("=", 1)
    else:
        path = ""
        rhs = arg
    rhs_obj = parse_json_or_str(rhs)
    return merge_at(obj, path, rhs_obj)


def from_any(x: Any) -> JsonValue:
    """
    Convert a Python object to json. This does not require type hints,
    since we have the actual object available.
    """

    # primitives
    if type(x) in (str, bool, int, float, type(None)):
        return x

    if isinstance(x, Enum):
        return from_any(x.value)

    if isinstance(x, abc.Mapping):
        return {str_from_any(k): from_any(v) for (k, v) in x.items()}

    if dataclasses.is_dataclass(x):
        return from_any(dataclasses.asdict(x))

    # NamedTuple
    if hasattr(x, "_asdict"):
        return from_any(x._asdict())

    if isinstance(x, (abc.Sequence, abc.Set)):
        return [from_any(x) for x in x]

    # NumPy floating-point types
    if isinstance(x, np.floating):
        return float(x)

    # NumPy integer types
    if isinstance(x, np.integer):
        return int(x)

    # Numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    raise ValueError(f"conversion of {x!r} to json not supported")


def str_from_any(x: Any) -> str:
    k = from_any(x)
    if not isinstance(k, str):
        raise TypeError(f"use of {x!r} as a json key is not supported")
    return k


def save_json(path: str, content: Any, extra_info: Dict = None) -> None:
    """Saves a content to a JSON file.

    Args:
        path: The path to the output JSON file.
        content: The content to save (typically a dictionary).
    """

    with open(path, "w", encoding="utf-8") as f:
        content_json = from_any(content)
        if extra_info:
            content_json.update(extra_info)
        json.dump(content_json, f, indent=2)


def load_json(path: str, keys_to_int: bool = False) -> Any:
    """Loads the content of a JSON file.

    Args:
        path: The path to the input JSON file.
        keys_to_int: Whether to convert keys to integers.
    Returns:
        The loaded content (typically a dictionary).
    """

    def convert_keys_to_int(x):
        return {
            int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()
        }

    with open(path, "r") as f:
        if keys_to_int:
            return json.load(f, object_hook=convert_keys_to_int)
        else:
            return json.load(f)


def _is_optional(t: Type) -> bool:
    # type: ignore
    return getattr(t, "__origin__", None) is Union and t.__args__[1:] == (
        type(None),
    )


def _is_sequence(t: Type) -> bool:
    """Returns true if T is a sequence-like type, e.g. List[int] or
    Set[int] or Tuple[int,...].
    """

    origin = getattr(t, "__origin__", None)
    # Python 3.7 has __origin__==List; python 3.8 changes to list
    if origin in (list, set, frozenset, List, Set, FrozenSet):
        return True
    if origin in (tuple, Tuple) and t.__args__[1:] == (Ellipsis,):
        return True
    return False


def get_real_type(t):
    """
    Get the actual type associated with a typing.Type.
    You can't do List[int]([1,2,3]); you have to do list([1,2,3]).

    __orig_bases__ lets us find the latter from the former.
    >>> get_real_type(List[int])
    <class 'list'>
    >>> get_real_type(Tuple[int, float])
    <class 'tuple'>
    """
    if hasattr(t, "__orig_bases__"):
        return t.__orig_bases__[0]
    else:
        return t.__origin__


def _append_path(path: str, key: Union[str, int]) -> str:
    "build up dotted path for error messages"
    if path:
        return f"{path[:-1]}.{key}'"
    else:
        return f" at '{key}'"


def validate_json(j: JsonValue, t: Type[T], at: str = "") -> T:
    """
    Convert a json object (one of None, bool, int, float, str, [json], {str: json})
    to an object of type t.

    t must be compatible type. Compatible types are:

    - NoneType, bool, int, float, src

    - typing.List/Set/FrozenSet/MutableSet[T] where T is a compatible type.

    - typing.Dict[str,V] where V is a compatible type

    - typing.Tuple[T1, T2, ...] where T1, T2, ... are compatible types. The
      tuple can end with an actual ellipsis to indicate a variable number of
      same-typed elements.

    - typing.NamedTuple(_, f1=T1, f2=T2, ...) where T1, T2, ... are compatible
      types.

    - Optional[T] where T is a compatible type.

    Raises TypeError if `t` isn't a compatible type.

    Raises ValueError if the json object doesn't match type t.
    """
    # type checkers just have no idea what's going on here, so delegate
    # to an unchecked function.
    return _validate_json(j, t, at)


def _validate_json(j, t, at: str):
    # unwrap optional types
    if _is_optional(t):
        if j is None:
            return None
        else:
            t = t.__args__[0]

    tname = getattr(t, "__name__", str(t))

    # bool, int and float interconvert as long as the value is unchanged.
    # So "3.0" is valid for an int and "4" is valid for a float, and "0"
    # and "1" are valid for bools.
    if t in (bool, int, float):
        if type(j) not in (bool, int, float) or t(j) != j:
            raise ValueError(f"{j!r} is not a valid value of type {tname}{at}")
        return t(j)

    # str and None must be exactly represented in json
    if t is str:
        if not isinstance(j, str):
            raise ValueError(f"expected json string but got {j}{at}")
        return j

    if t in (None, type(None)):
        if j is not None:
            raise ValueError(f"expected json null but got {j}{at}")
        return None

    origin = getattr(t, "__origin__", None)

    # sequence types
    if _is_sequence(t):
        return _json_to_seq(j, t, at)

    # tuples
    if origin in (tuple, Tuple):
        return _json_to_tuple(j, t, at)

    # mapping types
    if origin in (dict, Dict):
        return _json_to_dict(j, t, at)

    if origin is Union:
        for tt in t.__args__:
            try:
                return _validate_json(j, tt, at)
            except ValueError:
                pass
        raise ValueError(f"expected json Union but got {j}{at}")

    if inspect.isclass(t) and issubclass(t, Enum):
        return t(j)

    if hasattr(t, "__annotations__"):
        return _json_to_struct(j, t, at)

    # NumPy types
    if "numpy" in str(t):
        if np.issubdtype(t, np.ndarray):
            return np.array(j)
        else:
            return t(j)

    # If the passed-in type is not supported, that's a TypeError bug, not a ValueError
    # in the input.
    raise TypeError(f"don't know how to validate {t}{at}")


def _json_to_struct(j, t, at):
    """
    Convert json to a dataclass or namedtuple type.

    >>> class Named(NamedTuple):
    ...    a : int
    ...    b : float
    ...    c : Optional[int]
    ...    d : int = 4

    >>> _json_to_struct({'a': 1, 'b': 2, 'c': 3, 'd': 5}, Named, '')
    Named(a=1, b=2.0, c=3, d=5)
    >>> _json_to_struct({'a': 1, 'b': 2}, Named, '')
    Named(a=1, b=2.0, c=None, d=4)
    """
    if not isinstance(j, dict):
        raise ValueError(f"expected json dict but got {j!r}{at}")

    types = get_type_hints(t)
    args = {}
    for k, v in j.items():
        at_k = _append_path(at, k)

        if k not in types:
            raise ValueError(f"unknown field{at_k} for {t.__name__}")

        args[k] = _validate_json(v, types[k], at_k)

    try:
        return t(**args)
    except TypeError as e:
        # convert to ValueError and add context
        raise ValueError(str(e) + at) from e


def _json_to_tuple(j, tuple_type, at: str):
    """
    Convert json to a tuple type (like Tuple[int,float,str]).

    >>> _json_to_tuple([1, 2, 'x'], Tuple[int,float,str], '')
    (1, 2.0, 'x')
    >>> _json_to_tuple([1, 2, 3], Tuple[int,...], '')
    (1, 2.0, None)
    """
    elem_types = tuple_type.__args__

    if elem_types[1:] == [Ellipsis]:
        # sequence-like tuple
        return tuple(_json_to_seq(j, List[elem_types[0]], at))

    if not isinstance(j, (list, tuple)):
        raise ValueError(f"expected json list/tuple but got {j!r}{at}")

    n = len(elem_types)
    if len(j) != n:
        # fixed-size tuple
        raise ValueError(f"expected {n}-tuple but got {len(j)}{at}")

    return tuple(
        _validate_json(j[i], elem_types[i], _append_path(at, i))
        for i in range(n)
    )


def _json_to_seq(j, seq_type, at):
    """
    Convert json to a Sequence type (like List[int], Set[str], etc.)

    >>> _json_to_seq([1, 2], List[float])
    [1.0, 2.0]
    >>> _json_to_seq([1, 2], Set[float])
    {1.0, 2.0}
    >>> _json_to_seq([1, 2], FrozenSet[float])
    frozenset({1.0, 2.0})
    >>> _json_to_seq([1, 2, 3], Tuple[float,...])
    (1.0, 2.0, 3.0)

    >>> _json_to_seq(1, List[float])
    Traceback (most recent call last):
        ...
    ValueError: expected json list but got <class 'int'> for typing.List[float]
    """
    if not isinstance(j, list):
        raise ValueError(f"expected json list but got {j!r} for {seq_type}")
    elem_type = seq_type.__args__[0]
    elems = [
        _validate_json(elem, elem_type, _append_path(at, i))
        for (i, elem) in enumerate(j)
    ]

    return get_real_type(seq_type)(elems)


def _json_to_dict(j, t, at: str):
    """
    Convert a json dict to a typed Python dict, by just mapping validate_json() over
    the values.

    >>> _json_to_dict({'x':1, 'y': 2}, Dict[str, float], '')
    {'x': 1.0, 'y': 2.0}

    >>> _json_to_dict({'x':1, 'y': 2}, Dict[int, float], '')
    Traceback (most recent call last):
        ...
    TypeError: Dict types must have string keys

    >>> _json_to_dict([], Dict[str, float], '')
    Traceback (most recent call last):
        ...
    ValueError: expected json object but got <class 'list'> for typing.Dict[str, float]

    >>> _json_to_dict({'x':1}, Dict[str, str], '')
    Traceback (most recent call last):
        ...
    ValueError: expected <class 'str'> but got <class 'int'> at 'x'
    """
    kt, vt = t.__args__
    if not isinstance(j, dict):
        raise ValueError(f"expected json object but got {j!r} for {t}{at}")

    d = {}
    for k, v in j.items():
        at_k = _append_path(at, k)
        d[_validate_json(k, kt, at_k)] = _validate_json(v, vt, at_k)
    return d


############### utils/config_util.py ###############
"""Utility functions for managing configuration options."""


import argparse
import logging
import re
import typing
from typing import (
    Any,
    Dict,
    Mapping,
    NamedTuple,
    NamedTupleMeta,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


def print_opts(opts: NamedTuple) -> None:
    """Prints options.

    Args:
        opts: Options to be printed.
    """

    separator = "-" * 80
    logger.info(separator)
    logger.info(f"Options {opts.__class__.__name__}:")
    logger.info(separator)
    for name, value in opts._asdict().items():
        logger.info(f"- {name}: {value}")


def load_opts_from_raw_dict(
    opts_raw: Dict[str, Any],
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    opts = {}
    for name, type_def in opts_types.items():
        opts[name] = validate_json(opts_raw[name], type_def)
    if optional_opts_types is not None:
        for name, type_def in optional_opts_types.items():
            if name in opts_raw:
                opts[name] = validate_json(opts_raw[name], type_def)
            else:
                opts[name] = type_def()
    return opts


def load_opts_from_json(
    path: str,
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Loads options from a JSON file.

    Args:
        path: The path to the input JSON file.
        opts_types: A mapping from names of the expected option sets to
            types of the option sets. Example:
            opts_types = {
                "model_opts": config.ModelOpts,
                "train_opts": config.TrainOpts,
            }
            optional_opts_types: A mapping from names of the optional options sets to
            types of the option sets. If not given, set to the type defaults. Example:
            optional_opts_types = {
                "data_opts": config.DataOpts,
            }
    Returns:
        A dictionary mapping names of option sets to validated option sets.
    """

    opts_raw = load_json(path)
    return load_opts_from_raw_dict(opts_raw, opts_types, optional_opts_types)


def load_from_file(
    path: str,
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Loads options from a JSON or YAML file.
    Args:
        path: The path to the input JSON or YAML file.
        opts_types: A mapping from names of the expected option sets to
            types of the option sets.
        optional_opts_types: A mapping from names of the optional options sets to
            types of the option sets. If not given, set to the type defaults.
    Returns:
        A dictionary mapping names of option sets to validated option sets.
    """
    if path.endswith(".json"):
        return load_opts_from_json(path, opts_types, optional_opts_types)
    elif path.endswith(".yaml"):
        return load_opts_from_yaml(path, opts_types, optional_opts_types)
    else:
        raise ValueError(f"File {path} must be a .json or .yaml file")


def convert_to_parser_type(data_type: Any) -> Dict[str, Any]:
    """Converts a data type to a type description understood by ArgumentParser.

    Args:
        data_type: A data type.
    Returns:
        A dictionary with a description of the data type for ArgumentParser.
    """

    # List or Tuple (e.g., List[str], List[int]).
    if typing.get_origin(data_type) in [list, tuple]:
        return {"type": typing.get_args(data_type)[0], "nargs": "+"}

    # Boolean.
    elif data_type == bool:
        return {"type": lambda x: (str(x).lower() in ["true", "1"])}

    # Other types (see https://docs.python.org/3/library/argparse.html#type for
    # a list of supported types).
    else:
        return {"type": data_type}


def add_opts_to_parser(
    opts_type: NamedTuple, parser: argparse.ArgumentParser
) -> None:
    """Adds options of specified types and defaults to an ArgumentParser.

    Args:
        opts_type: A NamedTuple definition describing the options.
        parser: A parser to which the options are added.
    """

    # Iterate over fields of the NamedTuple definition, collect their types and
    # default values, and add them as options (aka arguments) to the parser.
    for field in opts_type._fields:
        field_info = {}

        # The default value.
        # pylint: disable=W0212, needs to access protected field _field_defaults
        if field in opts_type._field_defaults:
            # pylint: disable=W0212, needs to access protected field _field_defaults
            field_info["default"] = opts_type._field_defaults[field]

        # The data type.
        # pylint: disable=W0212, needs to access protected field _field_types
        if field in opts_type.__annotations__:
            # pylint: disable=W0212, needs to access protected field _field_types
            field_type = opts_type.__annotations__[field]

            # Optional fields (e.g., Optional[int], Optional[List[str]]) act as
            # Union[type, None] and need a special treatment.
            if typing.get_origin(field_type) == Union:
                type_args = typing.get_args(field_type)
                if len(type_args) > 2 or type_args[1] is not None.__class__:
                    raise ValueError(
                        "Only unions of a form Union[type, None] are supported."
                    )
                field_info.update(convert_to_parser_type(type_args[0]))
            else:
                field_info.update(convert_to_parser_type(field_type))

        # Add the option to the parser.
        field_name = field.replace("_", "-")
        parser.add_argument(f"--{field_name}", **field_info)


def parse_opts_from_command_line(
    opts_type: Union[NamedTuple, Mapping[str, NamedTuple]]
) -> Tuple[NamedTuple, Optional[str]]:
    """Parses options from the command line.

    Args:
        opts_type: A data structure defining the options to parse, or a dictionary
            of data structures. In the latter case, each dictionary item defines
            a sub-command and its options (see, e.g., assets.py for an example).
    Returns:
        A tuple with the parsed options and the subcommand name (None if a
        subcommand was specified).
    """

    # Create a parser of command-line arguments.
    parser = argparse.ArgumentParser()

    # Options specific to the selected sub-command.
    if isinstance(opts_type, Mapping):
        # Parse the options (a special parser is created for each sub-command).
        subparsers = parser.add_subparsers(dest="subcmd")
        for subcmd, subcmd_opts_type in opts_type.items():
            subcmd_parser = subparsers.add_parser(subcmd)
            add_opts_to_parser(subcmd_opts_type, subcmd_parser)
        args = parser.parse_args()

        # Check that a valid sub-command was provided.
        if args.subcmd not in opts_type.keys():
            raise ValueError(
                f"A subcommand required, one of: {opts_type.keys()}"
            )

        # Convert the parsed options to a NamedTuple.
        args_items = args.__dict__.items()
        opts = opts_type[args.subcmd](
            **{k: v for k, v in args_items if v is not None and k != "subcmd"}
        )

        return opts, args.subcmd

    # A single set of options (no sub-commands).
    else:
        # Parse the options.
        add_opts_to_parser(opts_type, parser)
        args = parser.parse_args()

        # Convert the parsed options to a NamedTuple.
        args_items = args.__dict__.items()
        opts = opts_type(**{k: v for k, v in args_items if v is not None})

        return opts, None


def camel_to_snake_name(name: str) -> str:
    """Convert a camel case name to a snake case name.

    Args:
        name: A camel case name (e.g. "InferOpts").
    Returns:
        A snake case name (e.g. "infer_opts").
    """

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_opts_from_json_or_command_line(
    opts_type: Union[NamedTuple, Mapping[str, NamedTuple]]
) -> Tuple[NamedTuple, Optional[str]]:
    """Loads options from a JSON file or the command line.

    The options are loaded from a JSON file specified via `--opts-path`
    command line argument. If this argument is not provided, then the
    options are read directly from the command line.

    Returns:
        A tuple with the parsed options and the subcommand name (None if a
        subcommand was specified).
    """

    # Try to parse argument `--opts-path`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--opts-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_known_args()[0]

    # Load options from a JSON file if `--opts-path` is specified.
    if args.opts_path is not None:
        if isinstance(opts_type, Mapping):
            # See function `parse_opts_from_command_line` for more details
            # about subcommands mentioned in the exception message below.
            raise ValueError(
                "Subcommands are not supported when loading from a JSON "
                "file. Please provide a single definition of options."
            )

        # Get a snake-case version of the options name (e.g. "InferOpts"
        # is converted to "infer_opts").
        opts_name = camel_to_snake_name(opts_type.__name__)

        # Load the options from a JSON file specified by `--opts-path`.
        opts = load_opts_from_json(
            path=args.opts_path, opts_types={opts_name: opts_type}
        )[opts_name]

        return opts, args

    # Otherwise parse options from the command line.
    else:
        return parse_opts_from_command_line(opts_type)


import os.path as osp
import time

############### utils/renderer.py ###############
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


class RenderType(Enum):
    """The rendering type.

    COLOR: An RGB image.
    DEPTH: A depth image with the depth values in mm.
    NORMAL: A normal map with normals expressed in the camera space.
    MASK: A binary mask.
    """

    COLOR = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    MASK = "mask"


class PyrenderRasterizer(RendererBase):
    """The base class which all renderers should inherit."""

    def __init__(
        self,
        renderer_flags: int = pyrender_RenderFlags_None,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        self.renderer_flags = renderer_flags

        # Per-object scene (a special scene is created for each object model).
        self.object_scenes: Dict[int, pyrender.Scene] = {}

        self.object_meshes: Dict[int, trimesh.Trimesh] = {}

        self.renderer: Optional[pyrender.OffscreenRenderer] = None
        self.im_size: Optional[Tuple[int, int]] = None

        if model_path:
            self.model_path = model_path

    def get_object_model(
        self,
        obj_id: int,
        mesh_color: Optional[Color] = None,
        **kwargs: Any,
    ) -> trimesh.Trimesh:
        """Gets the object model.

        Args:
            obj_id: The object ID.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        # Load the object model.
        object_model_path = self.model_path.format(obj_id=obj_id)
        trimesh_model = trimesh.load(object_model_path)
        trimesh_model.vertices = trimesh_model.vertices / 1000.0

        # Color the model.
        if mesh_color:
            num_vertices = trimesh_model.vertices.shape[0]
            trimesh_model.visual = trimesh.visual.objects.create_visual(
                vertex_colors=np.tile(mesh_color, (num_vertices, 1)),
                mesh=trimesh_model,
            )

        if obj_id not in self.object_meshes:
            self.object_meshes[obj_id] = trimesh_model

        return trimesh_model

    def add_object_model(
        self,
        obj_id: int,
        model_path: str,
        mesh_color: Optional[Color] = None,
        **kwargs: Any,
    ) -> None:
        """Adds an object model to the renderer.

        Args:
            asset_key: The key of an asset to add to the renderer.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        if obj_id in self.object_scenes:
            return

        # Load the object model.
        if obj_id not in self.object_meshes:

            trimesh_model = trimesh.load(model_path)
            if isinstance(trimesh_model, trimesh.Scene):
                # load from glb, unit is meter
                try:
                    trimesh_model = trimesh_model.to_mesh()
                except Exception as e:
                    raise e("Maybe update trimesh>=4.7.4 can fix it.")
            else:
                trimesh_model.vertices = trimesh_model.vertices / 1000.0
            # Color the model.
            if mesh_color:
                num_vertices = trimesh_model.vertices.shape[0]
                trimesh_model.visual = trimesh.visual.objects.create_visual(
                    vertex_colors=np.tile(mesh_color, (num_vertices, 1)),
                    mesh=trimesh_model,
                )
            self.object_meshes[obj_id] = trimesh_model
        else:
            trimesh_model = self.object_meshes[obj_id]

        mesh = pyrender.Mesh.from_trimesh(trimesh_model)

        # Create a scene and add the model to the scene in the canonical pose.
        ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
        self.object_scenes[obj_id] = pyrender.Scene(
            bg_color=np.zeros(4), ambient_light=ambient_light
        )
        self.object_scenes[obj_id].add(mesh)

    def render_object_model(
        self,
        obj_id: int,
        camera_model_c2w: CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, ArrayData]:
        """Renders an object model in the specified pose.

        Args:
            camera_model_c2m: A camera model with the extrinsics set to a rigid
                transformation from the camera to the model frame.
            render_types: Types of images to render.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        # Create a scene for the object model if it does not exist yet.
        if obj_id not in self.object_scenes:
            self.add_object_model(obj_id)

        # Render the scene.
        return self._render_scene(
            scene_in_w=self.object_scenes[obj_id],
            camera_model_c2w=camera_model_c2w,
            render_types=render_types,
            return_tensors=return_tensors,
            debug=debug,
        )

    def render_meshes(
        self,
        meshes_in_w: Sequence[trimesh.Trimesh],
        camera_model_c2w: CameraModel,
        render_types: Sequence[RenderType],
        mesh_colors: Optional[Sequence[Color]] = None,
        return_tensors: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, ArrayData]:
        """Renders a list of meshes (see the base class)."""

        # Create a scene.
        ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
        scene = pyrender.Scene(
            bg_color=np.zeros(4), ambient_light=ambient_light
        )

        # Add meshes to the scene.
        for mesh_id, mesh in enumerate(meshes_in_w):
            # Scale the mesh from mm to m (expected by pyrender).
            mesh.vertices /= (1000.0,)

            # Color the mesh.
            if mesh_colors:
                num_vertices = mesh.vertices.shape[0]
                mesh.visual = trimesh.visual.objects.create_visual(
                    vertex_colors=np.tile(
                        mesh_colors[mesh_id], (num_vertices, 1)
                    ),
                    mesh=mesh,
                )

            # Add the mesh to the scene.
            scene.add(pyrender.Mesh.from_trimesh(mesh))

        # Render the scene.
        output = self._render_scene(
            scene_in_w=scene,
            camera_model_c2w=camera_model_c2w,
            render_types=render_types,
            return_tensors=return_tensors,
            debug=debug,
        )

        # Scale the meshes back to mm.
        for mesh in meshes_in_w:
            mesh.vertices *= 1000.0

        return output

    def _render_scene(
        self,
        scene_in_w: "pyrender.Scene",
        camera_model_c2w: CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        debug: bool = False,
    ) -> Dict[RenderType, ArrayData]:
        """Renders an object model in the specified pose (see the base class)."""

        times = {}
        times["init_renderer"] = time.time()

        # Create the renderer if it does not exist yet, else check if the
        # rendering size is the one for which the renderer was created.
        if self.renderer is None:
            self.im_size = (camera_model_c2w.width, camera_model_c2w.height)
            self.renderer = pyrender.OffscreenRenderer(
                self.im_size[0], self.im_size[1]
            )
        elif (
            self.im_size[0] != camera_model_c2w.width
            or self.im_size[1] != camera_model_c2w.height
        ):
            raise ValueError("All renderings must be of the same size.")

        times["init_renderer"] = time.time() - times["init_renderer"]
        times["init_scene"] = time.time()

        # OpenCV to OpenGL camera frame.
        trans_cv2gl = get_opencv_to_opengl_camera_trans()
        trans_c2w = camera_model_c2w.T_world_from_eye.dot(trans_cv2gl)

        # Convert translation from mm to m, as expected by pyrender.
        trans_c2w[:3, 3] *= 0.001

        # Camera for rendering.
        camera = pyrender.IntrinsicsCamera(
            fx=camera_model_c2w.f[0],
            fy=camera_model_c2w.f[1],
            cx=camera_model_c2w.c[0],
            cy=camera_model_c2w.c[1],
            znear=0.1,
            zfar=3000.0,
        )

        # Create a camera node.
        camera_node = pyrender.Node(camera=camera, matrix=trans_c2w)
        scene_in_w.add_node(camera_node)
        # Create light.
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=2.4,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )

        light_node = pyrender.Node(light=light, matrix=trans_c2w)

        scene_in_w.add_node(light_node)

        times["init_scene"] = time.time() - times["init_scene"]
        times["render"] = time.time()

        # Rendering.
        color = None
        depth = None
        if self.renderer_flags & pyrender.RenderFlags.DEPTH_ONLY:
            assert self.renderer is not None
            depth = self.renderer.render(scene_in_w, flags=self.renderer_flags)
        else:
            assert self.renderer is not None
            color, depth = self.renderer.render(
                scene_in_w, flags=self.renderer_flags
            )

        times["render"] = time.time() - times["render"]
        times["postprocess"] = time.time()

        # Convert the RGB image from [0, 255] to [0.0, 1.0].
        if color is not None:
            color = color.astype(np.float32) / 255.0

        # Convert the depth map to millimeters.
        if depth is not None:
            depth *= 1000.0

        # Get the object mask.
        mask = None
        if RenderType.MASK in render_types:
            mask = depth > 0

        # Remove the camera so the scene contains only the object in the
        # canonical pose and can be reused.
        scene_in_w.remove_node(camera_node)
        scene_in_w.remove_node(light_node)

        # Prepare the output.
        output = {
            RenderType.COLOR: color,
            RenderType.DEPTH: depth,
            RenderType.MASK: mask,
        }
        if return_tensors:
            for name in output.keys():
                if output[name] is not None:
                    output[name] = misc.array_to_tensor(output[name])

        times["postprocess"] = time.time() - times["postprocess"]

        if debug:
            logger.info("PyrenderRasterizer run times:")
            for time_name, time_value in times.items():
                logger.info(f"- {time_name}: {time_value:.04f}s")

        return output


def get_opengl_to_opencv_camera_trans() -> np.ndarray:
    """Returns a transformation from OpenGL to OpenCV camera frame.

    Returns:
        A 4x4 transformation matrix (flipping Y and Z axes).
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    return yz_flip


def get_opencv_to_opengl_camera_trans() -> np.ndarray:
    return get_opengl_to_opencv_camera_trans()


############### utils/renderer_builder.py ###############
"""Utilities for building renderers."""

from enum import Enum
from typing import Any, Optional

# from utils import renderer_base


class RendererType(Enum):
    """The renderer to be used."""

    PYRENDER_RASTERIZER = "pyrender_rasterizer"


def build(
    renderer_type: RendererType,
    **kwargs: Any,
) -> RendererBase:
    """Builds renderer given the render name.

    Args:
        renderer_type: Name of the renderer.
        asset_library: Optional asset library to initialize the renderer with.
    Returns:
        A model.
    """
    if renderer_type == RendererType.PYRENDER_RASTERIZER:
        # from utils import renderer
        return PyrenderRasterizer(**kwargs)
    else:
        raise ValueError(f"Unknown renderer `{renderer_type}`.")


############### gen_templates.py ###############

#!/usr/bin/env python3

"""Synthesizes object templates."""


import os
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import imageio
import numpy as np

# from bop_toolkit_lib import inout, dataset_params
# import bop_toolkit_lib.config as bop_config

## from utils import (
##     misc as foundpose_misc,
##     json_util,
##     config_util,
##     logging,
##     misc,
##     structs
## )
## from utils.structs import AlignedBox2f, PinholePlaneCameraModel
## from utils.misc import warp_depth_image, warp_image
## from utils import geometry, renderer_builder
## from utils.renderer_base import RenderType


class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str = "v1"
    object_dataset: str = "lmo"

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    # Mesh pre-processing options.
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    # Rendering options.
    ssaa_factor: float = 4.0
    background_type: str = "black"
    light_type: str = "multi_directional"

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    features_patch_size: int = 14
    save_templates: bool = True
    debug: bool = True


@otn_manager.NODE.register(name="cad-to-templates")
def cad_to_templates(
    model_path: str = None,
    output_dir: str = "tmp_rendering_output",
    crop: bool = False,
    num_viewspheres: int = 1,
    min_num_viewpoints: int = 57,
    num_inplane_rotations: int = 14,
    min_depth: float = 346.31,
    max_depth: float = 1499.84,
    azimuth_range: str = "(0, 2 * math.pi)",
    elev_range: str = "(-0.5 * math.pi, 0.5 * math.pi)",
    topk: int = -1,
) -> None:
    """Reneder images (color, mask, depth) around object.
    Mitigrated from FoundPose(https://github.com/facebookresearch/foundpose)
    """
    opts = GenTemplatesOpts()

    # update opts from args
    opts = from_any(GenTemplatesOpts())
    opts["crop"] = crop
    opts["num_viewspheres"] = num_viewspheres
    opts["min_num_viewpoints"] = min_num_viewpoints
    opts["num_inplane_rotations"] = num_inplane_rotations
    opts_name = camel_to_snake_name(GenTemplatesOpts.__name__)
    opts: GenTemplatesOpts = load_opts_from_raw_dict(
        {opts_name: opts}, {opts_name: GenTemplatesOpts}
    )[opts_name]

    azimuth_range = eval(azimuth_range)
    elev_range = eval(elev_range)

    # Fix the random seed for reproducibility.
    np.random.seed(0)

    # Prepare a logger and a timer.
    logger = get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = Timer(enabled=opts.debug)
    timer.start()

    datasets_path = os.getenv("BOP_DATASET_PATH", "~/data/bop_datasets")
    datasets_path = os.path.expanduser(datasets_path)

    # Get properties of the default camera for the specified dataset.
    camera_json = os.path.join(
        datasets_path, opts.object_dataset, "camera.json"
    )
    c = json.load(open(camera_json, "r"))
    bop_camera = {
        "cam_params_path": camera_json,
        "im_size": (c["width"], c["height"]),
        "K": np.array(
            [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]]
        ),
    }
    # bop_camera = dataset_params.get_camera_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
    logger.info(f"Bop camera details are read ")

    # print("Bop camera params: \n", bop_camera)

    # Prepare a camera for the template (square viewport of a size divisible by the patch size).
    bop_camera_width = bop_camera["im_size"][0]
    bop_camera_height = bop_camera["im_size"][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = opts.features_patch_size * int(
        max_image_side / opts.features_patch_size
    )
    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera["K"][0, 0], bop_camera["K"][1, 1]),
        c=(
            bop_camera["K"][0, 2] - 0.5 * (bop_camera_width - image_side),
            bop_camera["K"][1, 2] - 0.5 * (bop_camera_height - image_side),
        ),
    )
    # Prepare a camera for rendering, upsampled for SSAA (supersampling anti-aliasing).
    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * opts.ssaa_factor),
        height=int(camera_model.height * opts.ssaa_factor),
        f=(
            camera_model.f[0] * opts.ssaa_factor,
            camera_model.f[1] * opts.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * opts.ssaa_factor,
            camera_model.c[1] * opts.ssaa_factor,
        ),
    )
    print("camera model created")

    # Build a renderer.
    render_types = [RenderType.COLOR, RenderType.DEPTH, RenderType.MASK]
    renderer_type = RendererType.PYRENDER_RASTERIZER
    renderer = build(renderer_type=renderer_type)

    # Define radii of the view spheres on which we will sample viewpoints.
    # The specified number of radii is sampled uniformly in the range of
    # camera-object distances from the test split of the specified dataset.
    # depth_range = (346.31, 1499.84) # from bop_test_split
    # min_depth = np.min(depth_range)
    # max_depth = np.max(depth_range)
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(opts.num_viewspheres)
    viewsphere_radii = []
    for depth_cell_id in range(opts.num_viewspheres):
        viewsphere_radii.append(
            min_depth + (depth_cell_id + 0.5) * depth_cell_size
        )

    # Generate viewpoints from which the object model will be rendered.
    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=radius,
            azimuth_range=azimuth_range,
            elev_range=elev_range,
            mode="fibonacci",
        )[0]
    logger.info(f"Sampled points on the sphere: {len(views_sphere)}")

    # Add in-plane rotations.
    if opts.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / opts.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(opts.num_inplane_rotations):
                R_inplane = rotation_matrix_numpy(
                    inplane_angle * inplane_id, np.array([0, 0, 1])
                )[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )
    logger.info(f"Number of views: {len(views)}")

    timer.elapsed("Time for setting up the stage")

    # Generate templates for each specified object.
    object_lid = 0

    log_heading(logger, f"Object {object_lid} from {opts.object_dataset}")
    timer.start()

    print("output_dir: ", output_dir)
    if os.path.exists(output_dir):
        print(f"remove cache {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")

    # Save parameters to a file.
    config_path = os.path.join(output_dir, "config.json")
    extra_info = {"model_path": model_path}
    save_json(config_path, opts, extra_info)

    # Prepare folder for saving templates.
    templates_rgb_dir = os.path.join(output_dir, "rgb")
    if opts.save_templates:
        os.makedirs(templates_rgb_dir, exist_ok=True)

    templates_depth_dir = os.path.join(output_dir, "depth")
    if opts.save_templates:
        os.makedirs(templates_depth_dir, exist_ok=True)

    templates_mask_dir = os.path.join(output_dir, "mask")
    if opts.save_templates:
        os.makedirs(templates_mask_dir, exist_ok=True)

    # Add the model to the renderer.
    renderer.add_object_model(
        obj_id=object_lid, model_path=model_path, debug=True
    )

    # Prepare a metadata list.
    metadata_list = []

    timer.elapsed("Time for preparing object data")

    template_list = []
    template_counter = 0
    for view_id, view in enumerate(views):
        logger.info(
            f"Rendering object {object_lid} from {opts.object_dataset}, view {view_id}/{len(views)}..."
        )
        for _ in range(opts.images_per_view):

            processed_num = len(os.listdir(templates_rgb_dir))
            if processed_num > template_counter:
                template_counter += 1
                continue

            if topk > 0 and template_counter >= topk:
                break

            timer.start()

            # Transformation from model to camera and inverse
            trans_m2c = RigidTransform(R=view["R"], t=view["t"])
            trans_c2m = trans_m2c.inv()

            # Camera model for rendering.
            trans_c2m_matrix = get_rigid_matrix(trans_c2m)
            render_camera_model_c2w = PinholePlaneCameraModel(
                width=render_camera_model.width,
                height=render_camera_model.height,
                f=render_camera_model.f,
                c=render_camera_model.c,
                T_world_from_eye=trans_c2m_matrix,
            )

            # Rendering.
            output = renderer.render_object_model(
                obj_id=object_lid,
                camera_model_c2w=render_camera_model_c2w,
                render_types=render_types,
                return_tensors=False,
                debug=False,
            )

            # Convert rendered mask.
            if RenderType.MASK in output:
                output[RenderType.MASK] = (
                    255 * output[RenderType.MASK]
                ).astype(np.uint8)

            # Calculate 2D bounding box of the object and make sure
            # it is within the image.
            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            if (
                object_box.left == 0
                or object_box.top == 0
                or object_box.right == render_camera_model_c2w.width - 1
                or object_box.bottom == render_camera_model_c2w.height - 1
            ):
                raise ValueError("The model does not fit the viewport.")

            # Optionally crop the object region.
            if opts.crop:
                # Get box for cropping.
                crop_box = calc_crop_box(
                    box=object_box,
                    make_square=True,
                )

                # Construct a virtual camera focused on the box.
                crop_camera_model_c2w = construct_crop_camera(
                    box=crop_box,
                    camera_model_c2w=render_camera_model_c2w,
                    viewport_size=(
                        int(opts.crop_size[0] * opts.ssaa_factor),
                        int(opts.crop_size[1] * opts.ssaa_factor),
                    ),
                    viewport_rel_pad=opts.crop_rel_pad,
                )

                # Map the images to the virtual camera.
                for output_key in output.keys():
                    if output_key in [RenderType.DEPTH]:
                        output[output_key] = warp_depth_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_depth_image=output[output_key],
                        )
                    elif output_key in [RenderType.COLOR]:
                        interpolation = (
                            cv2.INTER_AREA
                            if crop_box.width >= crop_camera_model_c2w.width
                            else cv2.INTER_LINEAR
                        )
                        output[output_key] = warp_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_image=output[output_key],
                            interpolation=interpolation,
                        )
                    else:
                        output[output_key] = warp_image(
                            src_camera=render_camera_model_c2w,
                            dst_camera=crop_camera_model_c2w,
                            src_image=output[output_key],
                            interpolation=cv2.INTER_NEAREST,
                        )

                # The virtual camera is becoming the main camera.
                camera_model_c2w = crop_camera_model_c2w.copy()
                scale_factor = opts.crop_size[0] / float(
                    crop_camera_model_c2w.width
                )
                camera_model_c2w.width = opts.crop_size[0]
                camera_model_c2w.height = opts.crop_size[1]
                camera_model_c2w.c = (
                    camera_model_c2w.c[0] * scale_factor,
                    camera_model_c2w.c[1] * scale_factor,
                )
                camera_model_c2w.f = (
                    camera_model_c2w.f[0] * scale_factor,
                    camera_model_c2w.f[1] * scale_factor,
                )

            # In case we are not cropping.
            else:
                camera_model_c2w = PinholePlaneCameraModel(
                    width=camera_model.width,
                    height=camera_model.height,
                    f=camera_model.f,
                    c=camera_model.c,
                    T_world_from_eye=trans_c2m_matrix,
                )

            # Downsample the renderings to the target size in case of SSAA.
            if opts.ssaa_factor != 1.0:
                target_size = (camera_model_c2w.width, camera_model_c2w.height)
                for output_key in output.keys():
                    if output_key in [RenderType.COLOR]:
                        interpolation = cv2.INTER_AREA
                    else:
                        interpolation = cv2.INTER_NEAREST

                    output[output_key] = resize_image(
                        image=output[output_key],
                        size=target_size,
                        interpolation=interpolation,
                    )

            # Record the template in the template list.
            template_list.append(
                {
                    "seq_id": template_counter,
                }
            )

            # Model and world coordinate frames are aligned.
            trans_m2w = RigidTransform(R=np.eye(3), t=np.zeros((3, 1)))

            # The object is fully visible.
            visibility = 1.0

            # Recalculate the object bounding box (it changed if we constructed the virtual camera).
            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(calc_2d_box(xs, ys))
            object_box = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            rgb_image = np.asarray(255.0 * output[RenderType.COLOR], np.uint8)
            depth_image = output[RenderType.DEPTH]

            # Object annotation.
            # object_anno = ObjectAnnotation(
            #     dataset=opts.object_dataset,
            #     lid=object_lid,
            #     pose=trans_m2w,
            #     boxes_amodal=np.array([object_box.array_ltrb()]),
            #     masks_modal=np.array([output[RenderType.MASK]], dtype=np.uint8),
            #     visibilities=np.array([visibility]),
            # )

            # Create a FrameSequence and write it to the Torch dataset.
            # data: Dict[str, Any] = dataset_util.pack_frame_sequence(
            #     sequence=FrameSequence(
            #         num_frames=1,
            #         num_views=1,
            #         images=np.array([[rgb_image]]),
            #         depth_images=np.array([[depth_image]]),
            #         cameras=[[camera_model_c2w]],
            #         frames_anno=frames_anno,
            #         objects_anno=[[object_anno]],
            #     ),
            # )

            timer.elapsed("Time for template generation")

            # Save template rgb, depth and mask.
            timer.start()
            rgb_path = os.path.join(
                templates_rgb_dir, f"template_{template_counter:04d}.png"
            )
            logger.info(
                f"Saving template RGB {template_counter} to: {rgb_path}"
            )
            # inout.save_im(rgb_path, rgb_image)
            imageio.imwrite(rgb_path, rgb_image)

            depth_path = os.path.join(
                templates_depth_dir, f"template_{template_counter:04d}.png"
            )
            logger.info(
                f"Saving template depth map {template_counter} to: {depth_path}"
            )

            # inout.save_depth(depth_path, depth_image)
            depth_image_uint16 = np.round(depth_image).astype(np.uint16)
            imageio.imwrite(depth_path, depth_image_uint16)

            # Save template mask.
            mask_path = os.path.join(
                templates_mask_dir, f"template_{template_counter:04d}.png"
            )
            logger.info(
                f"Saving template binary mask {template_counter} to: {mask_path}"
            )
            # inout.save_im(mask_path, output[RenderType.MASK])
            imageio.imwrite(mask_path, output[RenderType.MASK])

            data = {
                "dataset": opts.object_dataset,
                "lid": object_lid,
                "template_id": template_counter,
                "pose": trans_m2w,
                "boxes_amodal": np.array([object_box.array_ltrb()]).tolist(),
                "visibilities": np.array([visibility]).tolist(),
                "cameras": camera_model_c2w.to_json(),
                "rgb_image_path": rgb_path,
                "depth_map_path": depth_path,
                "binary_mask_path": mask_path,
            }
            timer.elapsed("Time for template saving")

            metadata_list.append(data)

            template_counter += 1

    # Save the metadata to be read from object repre.
    metadata_path = os.path.join(output_dir, "metadata.json")
    save_json(metadata_path, metadata_list)


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


@otn_manager.NODE.register(name="templates-to-linemod")
def templates_to_linemod(
    input_path: str,
    output_path: str,
    category: str = "object",
    topk: int = -1,
    save_depth: bool = False,
    background_imgdir: str = "/mnt/d/github/data/VOCdevkit/VOC2012/JPEGImages",
    cache: bool = False,
) -> bool:
    """
    Convert FoundPose template rendering output to LineMOD format.

    Args:
        input_path: Path to FoundPose rendering output directory
        output_path: Output directory for LineMOD format dataset
        category: Object category name (default: "object")
        topk: Maximum number of samples to process (default: 500)

    Returns:
        bool: True if successful, False otherwise
    """
    input_path = Path(input_path)
    output_path = Path(output_path) / category

    if output_path.exists() and not cache:
        shutil.rmtree(output_path)

    # Verify source directory structure
    rgb_dir = input_path / "rgb"
    mask_dir = input_path / "mask"
    depth_dir = input_path / "depth"
    config_file = input_path / "config.json"
    meta_file = input_path / "metadata.json"

    if not rgb_dir.exists():
        typer.echo(
            typer.style(
                f"Error: RGB directory {rgb_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if not mask_dir.exists():
        typer.echo(
            typer.style(
                f"Error: Mask directory {mask_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if save_depth and not depth_dir.exists():
        typer.echo(
            typer.style(
                f"Error: Depth directory {depth_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if not config_file.exists():
        typer.echo(
            typer.style(
                f"Error: Config file {config_file} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if not meta_file.exists():
        typer.echo(
            typer.style(
                f"Error: Meta data file {meta_file} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "JPEGImages").mkdir(exist_ok=True)
    (output_path / "mask").mkdir(exist_ok=True)
    (output_path / "labels").mkdir(exist_ok=True)
    if save_depth:
        (output_path / "depth").mkdir(exist_ok=True)

    # Load config
    with open(config_file, "r") as f:
        config = json.load(f)

    with open(meta_file, "r") as f:
        meta_data = json.load(f)
        meta_data = sorted(meta_data, key=lambda x: x["rgb_image_path"])

    model_path = config.get("model_path")
    target_path = output_path / (category + os.path.splitext(model_path)[1])
    shutil.copy2(model_path, target_path)

    vertices = load_points_3d_from_cad(
        model_path, src_unit="meter", dst_unit="millimeter"
    )
    bbox_3d = get_3D_corners(vertices)
    diameter = calc_pts_diameter(vertices)

    # Get list of template files
    rgb_files = sorted(list(rgb_dir.glob("template_*.png")))
    mask_files = sorted(list(mask_dir.glob("template_*.png")))
    if save_depth:
        depth_files = sorted(list(depth_dir.glob("template_*.png")))
        if len(rgb_files) != len(depth_files):
            typer.echo(
                typer.style(
                    f"Warning: RGB files ({len(rgb_files)}) != Depth files ({len(depth_files)})",
                    fg=typer.colors.YELLOW,
                )
            )
    if len(rgb_files) != len(mask_files):
        typer.echo(
            typer.style(
                f"Warning: RGB files ({len(rgb_files)}) != Mask files ({len(mask_files)})",
                fg=typer.colors.YELLOW,
            )
        )

    # Limit number of samples
    if topk > 0:
        rgb_files = rgb_files[:topk]
        mask_files = mask_files[:topk]
        if save_depth:
            depth_files = depth_files[:topk]
        meta_data = meta_data[:topk]

    valid_samples = []

    typer.echo(f"Processing {len(rgb_files)} samples...")
    camera_fx, camera_fy, camera_u0, camera_v0 = None, None, None, None

    # Process each template
    for i, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
        try:
            # Load RGB image
            rgb_img = cv2.imread(str(rgb_file))
            if rgb_img is None:
                typer.echo(f"Warning: Could not load RGB image {rgb_file}")
                continue

            # Load mask image
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                typer.echo(f"Warning: Could not load mask image {mask_file}")
                continue

            # Get image dimensions
            h, w = rgb_img.shape[:2]

            # Extract filename number
            filename = rgb_file.stem  # e.g., "template_0000"
            template_num = filename.split("_")[-1]  # e.g., "0000"

            # Copy RGB image to JPEGImages
            dst_rgb = output_path / "JPEGImages" / f"{i:06d}.png"
            shutil.copy2(rgb_file, dst_rgb)

            if save_depth:
                dst_depth = output_path / "depth" / f"{i:06d}.png"
                shutil.copy2(depth_files[i], dst_depth)

            # Copy mask to mask directory (with different naming convention)
            dst_mask = output_path / "mask" / f"{i:04d}.png"
            # Convert mask to 3-channel format as expected by linemod
            mask_3ch = np.stack([mask_img, mask_img, mask_img], axis=2)
            cv2.imwrite(str(dst_mask), mask_3ch)

            # Load meta data
            assert os.path.basename(
                meta_data[i]["rgb_image_path"]
            ) == os.path.basename(rgb_file)
            img_h, img_w = rgb_img.shape[:2]
            assert img_h == meta_data[i]["cameras"]["ImageSizeY"]
            assert img_w == meta_data[i]["cameras"]["ImageSizeX"]
            fx = meta_data[i]["cameras"]["fx"]
            fy = meta_data[i]["cameras"]["fy"]
            cx = meta_data[i]["cameras"]["cx"]
            cy = meta_data[i]["cameras"]["cy"]
            if camera_fx is None:
                camera_fx, camera_fy, camera_u0, camera_v0 = fx, fy, cx, cy

            K = np.eye(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            T_obj_cam = np.asarray(
                meta_data[i]["cameras"]["T_WorldFromCamera"]
            )
            T_cam_obj = np.linalg.inv(T_obj_cam)
            bbox_cam = (T_cam_obj[:3, :3] @ bbox_3d.T).T + T_cam_obj[:3, 3:4].T
            bbox_homo = K @ bbox_cam.T
            bbox_2d = bbox_homo[:2, :] / bbox_homo[2:3, :]

            # from pdebug.visp import draw
            # vis_bbox1 = draw.object_pose(rgb_img, T_cam_obj[:3, :3], T_cam_obj[:3, 3:4], vertices, K, color=(0, 255, 255))
            # vis_bbox2 = draw.points(rgb_img, bbox_2d.T)
            # cv2.imwrite("aa.png", np.concatenate((vis_bbox1, vis_bbox2), axis=1))

            # Format: class_id + 9*2 keypoints (normalized) + bbox_w + bbox_h + camera_params
            label_lines = []

            # Find bounding box of mask
            mask_binary = mask_img > 0
            if np.any(mask_binary):
                y_coords, x_coords = np.where(mask_binary)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                bbox_w = (x_max - x_min) / w
                bbox_h = (y_max - y_min) / h

                # Create dummy 3D bbox corners (normalized to image coordinates)
                # Since we don't have 3D pose, we'll create a simple square pattern around the center
                bbox_size = min(bbox_w, bbox_h) * 0.5

                # 9 keypoints: center + 8 corners of 3D bbox
                keypoints = bbox_2d.astype(np.float32).T
                keypoints[:, 0] /= img_w
                keypoints[:, 1] /= img_h

                # Create label line
                label_parts = [0.0]  # class_id
                label_parts.extend(
                    keypoints.flatten()
                )  # 18 values for 9 keypoints
                label_parts.extend([bbox_w, bbox_h])  # bbox dimensions
                label_parts.extend(
                    [camera_fx, camera_fy, float(w), float(h)]
                )  # camera fx, fy, w, h
                label_parts.extend(
                    [camera_u0, camera_v0, float(w), float(h)]
                )  # camera u0, v0, w, h

                label_line = " ".join([f"{v:.6f}" for v in label_parts])
                label_lines.append(label_line)
            else:
                typer.echo(f"Warning: Empty mask for {mask_file}")
                continue

            # Write label file
            dst_label = output_path / "labels" / f"{i:06d}.txt"
            with open(dst_label, "w") as f:
                for line in label_lines:
                    f.write(line + "\n")

            valid_samples.append(f"{i:06d}.png")

            if (i + 1) % 50 == 0:
                typer.echo(f"Processed {i + 1}/{len(rgb_files)} samples...")

        except Exception as e:
            raise e
            typer.echo(f"Error processing sample {i}: {e}")
            continue

    if not valid_samples:
        typer.echo(
            typer.style(
                "Error: No valid samples processed", fg=typer.colors.RED
            )
        )
        return False

    # Create train/test split (80/20)
    np.random.shuffle(valid_samples)
    split_idx = int(0.8 * len(valid_samples))
    train_samples = sorted(valid_samples[:split_idx])
    test_samples = sorted(valid_samples[split_idx:])

    # Write train.txt
    train_file = output_path / "train.txt"
    with open(train_file, "w") as f:
        for sample in train_samples:
            f.write(sample + "\n")

    # Write test.txt
    test_file = output_path / "test.txt"
    with open(test_file, "w") as f:
        for sample in test_samples:
            f.write(sample + "\n")

    # Create camera intrinsic file
    camera_data = {
        "distortion": None,
        "intrinsic": [
            [camera_fx, 0.0, camera_u0],
            [0.0, camera_fy, camera_v0],
            [0.0, 0.0, 1.0],
        ],
    }
    camera_file = output_path / "linemod_camera.json"
    with open(camera_file, "w") as f:
        json.dump(camera_data, f, indent=2)

    # Create YAML config
    yaml_config = {
        "train": str(train_file.absolute()),
        "val": str(test_file.absolute()),
        "test": str(test_file.absolute()),
        "names": [category],
        "nc": 1,
        "fx": camera_fx,
        "fy": camera_fy,
        "u0": camera_u0,
        "v0": camera_v0,
        "diam": diameter,
        "mesh": model_path,
    }
    if background_imgdir:
        yaml_config["background_path"] = background_imgdir

    yaml_file = output_path / f"{category}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    # Print summary
    typer.echo(
        typer.style(
            f"Successfully converted FoundPose data to LineMOD format",
            fg=typer.colors.GREEN,
        )
    )
    typer.echo(f"  Output directory: {output_path}")
    typer.echo(f"  Total samples: {len(valid_samples)}")
    typer.echo(f"  Train samples: {len(train_samples)}")
    typer.echo(f"  Test samples: {len(test_samples)}")
    typer.echo(f"  Category: {category}")

    return True


def prepare_sample(root: str, rgb_file: str) -> SceneAnnotation:
    """Get sample from custom data root.
    Used by Foundpose infer.

    Args:
        root: data root, containing rgb, camera.json
        rgb_file: rgb file path to be processed
    """

    rgb = imageio.v3.imread(rgb_file)
    depth_file = Path(rgb_file.replace("rgb", "spatracker/depth"))
    if depth_file.exists():
        depth = imageio.v3.imread(depth_file).astype(np.float32)
    else:
        depth = None

    camera_json = os.path.join(root, "camera.json")
    with open(camera_json, "r") as fid:
        camera_data = json.load(fid)

    fx = camera_data["cam_K"][0][0]
    fy = camera_data["cam_K"][1][1]
    cx = camera_data["cam_K"][0][2]
    cy = camera_data["cam_K"][0][2]
    camera = PinholePlaneCameraModel(
        width=rgb.shape[1],
        height=rgb.shape[0],
        f=tuple([fx, fy]),
        c=tuple([cx, cy]),
    )
    return SceneAnnotation(
        image=rgb,
        depth_image=depth,
        camera=camera,
        objects_anno=None,
    )


def get_instances_for_pose_estimation(root: str, rgb_file: str):
    insts = []
    mask_file = rgb_file.replace("/rgb/", "/sam2/")
    assert os.path.exists(mask_file), f"{mask_file} not found."
    mask = imageio.v3.imread(mask_file) / 255
    bbox = binary_mask_to_bbox(mask)

    inst = {
        "input_box_amodal": bbox,  # x1y1x2y2
        "input_mask_modal": mask,  # 0-1 mask
        "gt_anno": None,
        "gt_iou": None,
        "time": 1.0,  # fake_time
    }
    insts.append(inst)
    return insts


@otn_manager.NODE.register(name="foundpose-to-linemod")
def foundpose_to_linemod(
    input_path: str,
    pred_path: str,
    model_path: str,
    output_path: str = "tmp_linemod",
    category: str = "object",
    topk: int = -1,
    save_depth: bool = False,
    background_imgdir: str = "/mnt/d/github/data/VOCdevkit/VOC2012/JPEGImages",
    cache: bool = False,
) -> bool:
    """
    Convert FoundPose inference result to LineMOD format.

    Args:
        input_path: Path to FoundPose rendering output directory
        output_path: Output directory for LineMOD format dataset
        category: Object category name (default: "object")
        topk: Maximum number of samples to process (default: 500)

    Returns:
        bool: True if successful, False otherwise
    """
    input_path = Path(input_path)
    output_path = Path(output_path) / category

    if output_path.exists() and not cache:
        shutil.rmtree(output_path)

    # Verify source directory structure
    rgb_dir = input_path / "rgb"
    mask_dir = input_path / "sam2"
    depth_dir = input_path / "spatracker/depth"
    camera_json = input_path / "camera.json"

    if not rgb_dir.exists():
        typer.echo(
            typer.style(
                f"Error: RGB directory {rgb_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if not mask_dir.exists():
        typer.echo(
            typer.style(
                f"Error: Mask directory {mask_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    if save_depth and not depth_dir.exists():
        typer.echo(
            typer.style(
                f"Error: Depth directory {depth_dir} not found",
                fg=typer.colors.RED,
            )
        )
        return False

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "JPEGImages").mkdir(exist_ok=True)
    (output_path / "mask").mkdir(exist_ok=True)
    (output_path / "labels").mkdir(exist_ok=True)
    if save_depth:
        (output_path / "depth").mkdir(exist_ok=True)

    target_path = output_path / (category + os.path.splitext(model_path)[1])
    shutil.copy2(model_path, target_path)

    vertices = load_points_3d_from_cad(
        model_path, src_unit="meter", dst_unit="millimeter"
    )
    bbox_3d = get_3D_corners(vertices)
    diameter = calc_pts_diameter(vertices)

    with open(pred_path, "r") as f:
        preds = json.load(f)

    with open(camera_json, "r") as fid:
        camera_data = json.load(fid)
    fx = camera_data["cam_K"][0][0]
    fy = camera_data["cam_K"][1][1]
    cx = camera_data["cam_K"][0][2]
    cy = camera_data["cam_K"][0][2]

    # Get list of template files
    rgb_files = Input(rgb_dir, name="imgdir").get_reader().imgfiles
    mask_files = Input(mask_dir, name="imgdir").get_reader().imgfiles
    if save_depth:
        depth_files = Input(depth_dir, name="imgdir").get_reader().imgfiles
        if len(rgb_files) != len(depth_files):
            typer.echo(
                typer.style(
                    f"Warning: RGB files ({len(rgb_files)}) != Depth files ({len(depth_files)})",
                    fg=typer.colors.YELLOW,
                )
            )
    if len(rgb_files) != len(mask_files):
        typer.echo(
            typer.style(
                f"Warning: RGB files ({len(rgb_files)}) != Mask files ({len(mask_files)})",
                fg=typer.colors.YELLOW,
            )
        )
    assert len(rgb_files) == len(
        preds
    ), f"ERROR: RGB files ({len(rgb_files)}) != pred lines ({len(preds)})"

    # Limit number of samples
    if topk > 0:
        rgb_files = rgb_files[:topk]
        mask_files = mask_files[:topk]
        if save_depth:
            depth_files = depth_files[:topk]

    valid_samples = []

    typer.echo(f"Processing {len(rgb_files)} samples...")
    camera_fx, camera_fy, camera_u0, camera_v0 = None, None, None, None

    # Process each template
    for i, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
        rgb_file = Path(rgb_file)
        try:
            # Load RGB image
            rgb_img = cv2.imread(str(rgb_file))
            if rgb_img is None:
                typer.echo(f"Warning: Could not load RGB image {rgb_file}")
                continue

            # Load mask image
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                typer.echo(f"Warning: Could not load mask image {mask_file}")
                continue

            # Get image dimensions
            h, w = rgb_img.shape[:2]

            # Extract filename number
            filename = rgb_file.stem  # e.g., "template_0000"
            template_num = filename.split("_")[-1]  # e.g., "0000"

            # Copy RGB image to JPEGImages
            dst_rgb = output_path / "JPEGImages" / f"{i:06d}.png"
            shutil.copy2(rgb_file, dst_rgb)

            if save_depth:
                dst_depth = output_path / "depth" / f"{i:06d}.png"
                shutil.copy2(depth_files[i], dst_depth)

            # Copy mask to mask directory (with different naming convention)
            dst_mask = output_path / "mask" / f"{i:04d}.png"
            # Convert mask to 3-channel format as expected by linemod
            mask_3ch = np.stack([mask_img, mask_img, mask_img], axis=2)
            cv2.imwrite(str(dst_mask), mask_3ch)

            # Load meta data
            img_h, img_w = rgb_img.shape[:2]
            if camera_fx is None:
                camera_fx, camera_fy, camera_u0, camera_v0 = fx, fy, cx, cy

            K = np.eye(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            pred = preds[i]
            assert Path(pred["img_id"]).stem == rgb_file.stem
            T_cam_obj = np.eye(4)
            T_cam_obj[:3, :3] = np.asarray(pred["R"])
            T_cam_obj[:3, 3:4] = np.asarray(pred["t"])
            bbox_cam = (T_cam_obj[:3, :3] @ bbox_3d.T).T + T_cam_obj[:3, 3:4].T
            bbox_homo = K @ bbox_cam.T
            bbox_2d = bbox_homo[:2, :] / bbox_homo[2:3, :]

            # from pdebug.visp import draw
            # vis_bbox1 = draw.object_pose(rgb_img, T_cam_obj[:3, :3], T_cam_obj[:3, 3:4], vertices, K, color=(0, 255, 255))
            # vis_bbox2 = draw.points(rgb_img, bbox_2d.T)
            # cv2.imwrite("aa.png", np.concatenate((vis_bbox1, vis_bbox2), axis=1))

            # Format: class_id + 9*2 keypoints (normalized) + bbox_w + bbox_h + camera_params
            label_lines = []

            # Find bounding box of mask
            mask_binary = mask_img > 0
            if np.any(mask_binary):
                y_coords, x_coords = np.where(mask_binary)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                bbox_w = (x_max - x_min) / w
                bbox_h = (y_max - y_min) / h

                # Create dummy 3D bbox corners (normalized to image coordinates)
                # Since we don't have 3D pose, we'll create a simple square pattern around the center
                bbox_size = min(bbox_w, bbox_h) * 0.5

                # 9 keypoints: center + 8 corners of 3D bbox
                keypoints = bbox_2d.astype(np.float32).T
                keypoints[:, 0] /= img_w
                keypoints[:, 1] /= img_h

                # Create label line
                label_parts = [0.0]  # class_id
                label_parts.extend(
                    keypoints.flatten()
                )  # 18 values for 9 keypoints
                label_parts.extend([bbox_w, bbox_h])  # bbox dimensions
                label_parts.extend(
                    [camera_fx, camera_fy, float(w), float(h)]
                )  # camera fx, fy, w, h
                label_parts.extend(
                    [camera_u0, camera_v0, float(w), float(h)]
                )  # camera u0, v0, w, h

                label_line = " ".join([f"{v:.6f}" for v in label_parts])
                label_lines.append(label_line)
            else:
                typer.echo(f"Warning: Empty mask for {mask_file}")
                continue

            # Write label file
            dst_label = output_path / "labels" / f"{i:06d}.txt"
            with open(dst_label, "w") as f:
                for line in label_lines:
                    f.write(line + "\n")

            valid_samples.append(f"{i:06d}.png")

            if (i + 1) % 50 == 0:
                typer.echo(f"Processed {i + 1}/{len(rgb_files)} samples...")

        except Exception as e:
            raise e
            typer.echo(f"Error processing sample {i}: {e}")
            continue

    if not valid_samples:
        typer.echo(
            typer.style(
                "Error: No valid samples processed", fg=typer.colors.RED
            )
        )
        return False

    # Create train/test split (80/20)
    np.random.shuffle(valid_samples)
    split_idx = int(0.8 * len(valid_samples))
    train_samples = sorted(valid_samples[:split_idx])
    test_samples = sorted(valid_samples[split_idx:])

    # Write train.txt
    train_file = output_path / "train.txt"
    with open(train_file, "w") as f:
        for sample in train_samples:
            f.write(sample + "\n")

    # Write test.txt
    test_file = output_path / "test.txt"
    with open(test_file, "w") as f:
        for sample in test_samples:
            f.write(sample + "\n")

    # Create camera intrinsic file
    camera_data = {
        "distortion": None,
        "intrinsic": [
            [camera_fx, 0.0, camera_u0],
            [0.0, camera_fy, camera_v0],
            [0.0, 0.0, 1.0],
        ],
    }
    camera_file = output_path / "linemod_camera.json"
    with open(camera_file, "w") as f:
        json.dump(camera_data, f, indent=2)

    # Create YAML config
    yaml_config = {
        "train": str(train_file.absolute()),
        "val": str(test_file.absolute()),
        "test": str(test_file.absolute()),
        "names": [category],
        "nc": 1,
        "fx": camera_fx,
        "fy": camera_fy,
        "u0": camera_u0,
        "v0": camera_v0,
        "diam": diameter,
        "mesh": model_path,
    }
    if background_imgdir:
        yaml_config["background_path"] = background_imgdir

    yaml_file = output_path / f"{category}.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    # Print summary
    typer.echo(
        typer.style(
            f"Successfully converted FoundPose data to LineMOD format",
            fg=typer.colors.GREEN,
        )
    )
    typer.echo(f"  Output directory: {output_path}")
    typer.echo(f"  Total samples: {len(valid_samples)}")
    typer.echo(f"  Train samples: {len(train_samples)}")
    typer.echo(f"  Test samples: {len(test_samples)}")
    typer.echo(f"  Category: {category}")

    return True


if __name__ == "__main__":
    import typer

    typer.run(cad_to_templates)
