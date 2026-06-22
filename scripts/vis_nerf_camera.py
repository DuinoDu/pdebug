#!/usr/bin/env python
import json
import random
from typing import List

from pdebug.data_types import (
    NerfCamera,
    Open3dCameraTrajectory,
    PointcloudTensor,
)
from pdebug.debug.nerf import vis_pose
from pdebug.utils.three_utils import threejs_pcd
from pdebug.visp.colormap import Colormap

import numpy as np
import typer

OBJ_INFO = {
    # nerf
    "chair": (True, 0.40, 0.0, 0.0, 0.0),
    "drums": (True, 0.40, 0.0, 0.0, 0.0),
    "ficus": (True, 0.40, 0.0, 0.0, 0.0),
    "hotdog": (True, 0.40, 0.0, 0.0, 0.0),
    "lego": (True, 0.40, 0.0, 0.0, 0.0),
    "materials": (True, 0.40, 0.0, 0.0, 0.0),
    "mic": (True, 0.40, 0.0, 0.0, 0.0),
    "ship": (True, 0.40, 0.0, 0.0, 0.0),
    # jittor comp
    "Car": (False, 0.200, 0.0, 0.0, 0.0),
    "Coffee": (False, 1.600, 0.0, 0.0, 0.0),
    "Easyship": (False, 0.460, 0.0, 0.0, 0.0),
    "Scar": (False, 0.105, 0.0, 0.0, 0.0),
    "Scarf": (False, 0.075, 0.0, 0.0, 0.0),
}


def nerf_matrix_to_ngp(pose, scale=0.33):
    """Convert nerf pose to ngp pose.

    For the fox dataset, 0.33 scales camera radius to ~ 2
    """
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def rescale_pose(pose, scale):
    """Rescale pose."""
    pose[:3, 3] *= scale
    return pose


def move_pose(pose, offset):
    """Move pose position."""
    pose[:3, 3] += offset
    return pose


def compute_scale_from_pose(pose, target_radius=2.0) -> float:
    """Compute scale from camera pose."""
    pose = np.asarray(pose)
    radius = np.linalg.norm(pose[:, :3, 3], axis=-1).mean()
    scale = target_radius / radius
    return scale


# @task(name="my-tool")
def main(
    input_jsons: List[str],
    port: int = typer.Option(6006, help="http server port"),
    vr: bool = typer.Option(False, help="in vr mode"),
    scale: float = typer.Option(1.0, help="scene scale, used in vr mode"),
    is_nerf: bool = typer.Option(False, help="If is nerf, do pose convert"),
    sample: int = typer.Option(-1, help="random sample camera pose"),
    topk: int = typer.Option(None, help="topk pose"),
    target_radius: float = typer.Option(
        None, help="target camera sphere radius, used to compute scale"
    ),
    input_pcd: str = typer.Option(None, help="extra pointcloud file"),
):
    """Visualize nerf-format camera pose."""
    typer.echo(typer.style(f"loading {input_jsons}", fg=typer.colors.GREEN))

    colormap = Colormap(len(input_jsons), hex_mode=True)
    convert_scale = 1.0
    offset = (0.0, 0.0, 0.0)

    all_poses = []
    all_colors = []
    for idx, input_json in enumerate(input_jsons):

        try:
            # tmp hard code, ignore me.
            obj_name = input_json.split("/")[-2]
            if obj_name in OBJ_INFO:
                is_nerf, convert_scale, *offset = OBJ_INFO[obj_name]
        except Exception as e:
            pass

        if NerfCamera.is_valid(input_json):
            typer.echo(typer.style(f"nerf camera", fg=typer.colors.GREEN))
            nerf_camera = NerfCamera.fromfile(input_json)
            poses = [
                np.array(f["transform_matrix"], dtype=np.float32)
                for f in nerf_camera.frames
            ]
        elif Open3dCameraTrajectory.is_valid(input_json):
            typer.echo(typer.style(f"open3d camera", fg=typer.colors.GREEN))
            o3d_camera = Open3dCameraTrajectory.fromfile(input_json)
            poses = [
                np.array(p["extrinsic"], dtype=np.float32)
                .reshape(4, 4)
                .transpose()
                for p in o3d_camera.parameters
            ]
        else:
            raise RuntimeError(f"Unvalid input json: {input_json}")

        if sample > 0:
            random.shuffle(poses)
            poses = poses[:sample]

        if topk:
            poses = poses[:topk]

        if target_radius:
            convert_scale = compute_scale_from_pose(poses, target_radius)

        new_poses = []
        for pose in poses:
            if is_nerf:
                new_pose = nerf_matrix_to_ngp(pose, scale=convert_scale)
            else:
                new_pose = rescale_pose(pose, scale=convert_scale)
            new_pose = move_pose(new_pose, offset)
            new_poses.append(new_pose)
        poses = np.asarray(new_poses)
        all_poses.append(poses)
        all_colors.append(np.array([colormap[idx] for _ in range(len(poses))]))

    all_poses = np.concatenate(all_poses, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    for pose in all_poses:
        pass

    if input_pcd:
        pcd = PointcloudTensor.from_open3d(input_pcd).to_open3d()
        # pcd.translate(np.asarray([-5, 0, -7], dtype=np.float64))

        js_code = threejs_pcd(pcd, size=0.1)
    else:
        js_code = ""
    vis_pose(
        all_poses,
        colors=all_colors,
        port=port,
        vr=vr,
        scale=scale,
        js_code=js_code,
    )


if __name__ == "__main__":
    typer.run(main)
