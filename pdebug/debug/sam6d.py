import os
import time

from pdebug.data_types import Camera, PointcloudTensor, Tensor, x_to_ndarray
from pdebug.visp import Colormap
from pdebug.visp.rerun import rr_rgbd, rr_rgbd_blueprint

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

__all__ = ["pcd", "timestamp", "pcds"]

is_rr_init = False
rr_root = "world"
rr_idx = 0


def _init_rr(mode=""):
    global is_rr_init
    if is_rr_init:
        return
    ip = os.getenv("RERUN_IP", None)
    if not ip:
        raise ValueError("Please set RERUN_IP environment variable.")

    rr.init("Sam6d" + mode, spawn=False)
    rr.connect_grpc(f"rerun+http://{ip}:9876/proxy")

    blueprint = rrb.Tabs(
        rrb.Spatial3DView(
            name="object",
            origin=f"{rr_root}/object",
            contents=f"{rr_root}/object",
        ),
        rrb.Spatial3DView(
            name="match",
            origin=f"{rr_root}/match",
            contents=f"{rr_root}/match",
        ),
        rrb.Spatial3DView(
            name="concat",
            origin=f"{rr_root}/concat",
            contents=f"{rr_root}/concat",
        ),
        rrb.Spatial3DView(
            name="concat_coarse",
            origin=f"{rr_root}/concat_coarse",
            contents=f"{rr_root}/concat_coarse",
        ),
        rrb.Spatial3DView(
            name="concat_fine",
            origin=f"{rr_root}/concat_fine",
            contents=f"{rr_root}/concat_fine",
        ),
        rrb.Spatial3DView(
            name="whole_pts",
            origin=f"{rr_root}/whole_pts",
            contents=f"{rr_root}/whole_pts",
        ),
    )
    rr.send_blueprint(blueprint)
    rr.log(rr_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    is_rr_init = True


def should_debug():
    return os.getenv("DEBUG_SAM6D", "0") == "1"


def timestamp(mode=""):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)

    global rr_idx
    rr.set_time_seconds("time", rr_idx)
    rr_idx += 1


def whole_pts(data, mode=""):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)
    data = x_to_ndarray(data)
    rr.log(rr_root + "/whole_pts", rr.Points3D(data))


def pcd(data, name, color, batch_idx=0, mode="", normalize=False):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)
    data = x_to_ndarray(data)
    if data.ndim == 3:
        data = data[batch_idx]
    if normalize:
        data -= np.mean(data, axis=0)[None, :]

    color = np.asarray(color)
    assert color.ndim == 1
    color = np.repeat(color[None, :], data.shape[0], 0)
    rr.log(rr_root + f"/{name}", rr.Points3D(data, colors=color))


def pcds(data, name, color, batch_idx=0, mode="", normalize=True):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)

    assert isinstance(data, list)
    assert isinstance(color, list)
    assert len(data) == len(color)

    points = []
    colors = []
    for data_i, color_i in zip(data, color):
        data_i = x_to_ndarray(data_i)
        if data_i.ndim == 3:
            data_i = data_i[batch_idx]
        if normalize:
            data_i -= np.mean(data_i, axis=0)[None, :]
        points.append(data_i)
        color_i = np.asarray(color_i)
        color = np.repeat(color_i[None, :], data_i.shape[0], 0)
        colors.append(color)
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    rr.log(rr_root + f"/{name}", rr.Points3D(points, colors=colors))
