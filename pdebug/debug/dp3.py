import os
import time

from pdebug.data_types import Camera, PointcloudTensor, Tensor, x_to_ndarray
from pdebug.visp import Colormap
from pdebug.visp.rerun import rr_rgbd, rr_rgbd_blueprint

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

__all__ = ["obs", "act", "timestamp"]

is_rr_init = False
rr_root = "world"
rr_idx = 0

robotwin_motors = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def _init_rr(mode="", motors=robotwin_motors):
    global is_rr_init
    if is_rr_init:
        return
    ip = os.getenv("RERUN_IP", None)
    if not ip:
        raise ValueError("Please set RERUN_IP environment variable.")

    colormap = Colormap(len(motors))

    rr.init("3D-Diffusion-Policy" + mode, spawn=False)
    rr.connect_grpc(f"rerun+http://{ip}:9876/proxy")

    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D", origin=rr_root),
            rrb.TimeSeriesView(
                name="AgentPos",
                origin="/agent_pos",
                overrides={
                    f"/agent_pos/{motors[i]}": rr.SeriesLines.from_fields(
                        colors=colormap[i], names=motors[i]
                    )
                    for i in range(len(motors))
                },
            ),
            rrb.TimeSeriesView(
                name="Action",
                origin="/action",
                overrides={
                    f"/action/{motors[i]}": rr.SeriesLines.from_fields(
                        colors=colormap[i], names=motors[i]
                    )
                    for i in range(len(motors))
                },
            ),
            rrb.TimeSeriesView(
                name="Loss",
                origin="/loss",
                overrides={
                    f"/loss": rr.SeriesLines.from_fields(
                        colors=[0, 255, 255], names="loss"
                    )
                },
            ),
            row_shares=[7, 3],
        ),
        # rr_rgbd_blueprint(rr_root, "camera"),
        column_shares=[2, 1],
    )
    rr.send_blueprint(blueprint)
    rr.log(rr_root, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    is_rr_init = True


def should_debug():
    return os.getenv("DEBUG_DP3", "0") == "1"


def timestamp(mode=""):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)

    global rr_idx
    rr.set_time_seconds("time", rr_idx)
    rr_idx += 1


def obs(observation, batch_idx=0, act=None, mode="", motors=robotwin_motors):
    """
    obs is dict with keys:
        agent_pos
        point_cloud/pointcloud
        image
        depth
    """
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode)

    if "pointcloud" in observation:
        pcd = x_to_ndarray(observation["pointcloud"])
    else:
        pcd = x_to_ndarray(observation["point_cloud"])

    if pcd.ndim == 4:
        # pcd is batch, training mode, choose first sample in batch
        pcd = pcd[batch_idx]

    if pcd.ndim == 2:
        pcd = [pcd]

    for idx in range(len(pcd)):
        pcd_i = pcd[idx]
        assert pcd_i.ndim == 2
        rr.log(f"{rr_root}", rr.Points3D(pcd_i[:, :3], colors=pcd_i[:, 3:]))

        if "agent_pos" in observation:
            action = x_to_ndarray(observation["agent_pos"])
            if action.ndim == 3:
                action = action[batch_idx]
            elif action.ndim == 1:
                action = np.asarray([action])
            else:
                pass
            assert action.ndim == 2

            assert len(action[idx]) == len(motors)
            for action_i, action_name in enumerate(motors):
                rr.log(
                    f"agent_pos/{action_name}",
                    rr.Scalars(action[idx][action_i]),
                )

        rgb = depth = None
        if "image" in observation:
            image = x_to_ndarray(observation["image"][idx])
            assert image.ndim == 3
            rgb = np.moveaxis(image, 0, -1)
        if "depth" in observation:
            depth = x_to_ndarray(observation["depth"][idx])
            assert depth.ndim == 2
        rr_rgbd(rgb=rgb, depth=depth)

        if idx < len(pcd) - 1:
            timestamp()


def act(action, batch_idx=0, mode="", motors=robotwin_motors):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode, motors=motors)

    action = x_to_ndarray(action)
    if action.ndim == 3:
        action = action[batch_idx]
    elif action.ndim == 1:
        action = np.asarray([action])

    assert action.ndim == 2
    for idx in range(len(action)):
        assert len(action[idx]) == len(motors)
        for action_i, action_name in enumerate(motors):
            rr.log(f"action/{action_name}", rr.Scalars(action[idx][action_i]))
        timestamp()


def loss(value, batch_idx=0, mode=""):
    if not should_debug():
        return
    if not is_rr_init:
        _init_rr(mode=mode, motors=motors)

    value = float(x_to_ndarray(value))
    rr.log(f"loss", rr.Scalars(value))
