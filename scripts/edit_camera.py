#!/usr/bin/env python
import json
import random
from typing import List, Optional

from pdebug.data_types import NerfCamera, Open3dCameraTrajectory

import numpy as np
import typer
from scipy.spatial.transform import Rotation as R


# @task(name="my-tool")
def main(
    input_json: str,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Edit camera pose, including offset, scale and rotation."""
    if not output:
        output = input_json[:-5] + "_update.json"

    if NerfCamera.is_valid(input_json):
        typer.echo(
            typer.style(f"camera type: nerf camera", fg=typer.colors.GREEN)
        )
        camera = NerfCamera.fromfile(input_json)
        poses = [
            np.array(f["transform_matrix"], dtype=np.float32)
            for f in camera.frames
        ]
    elif Open3dCameraTrajectory.is_valid(input_json):
        typer.echo(
            typer.style(f"camera type: open3d camera", fg=typer.colors.GREEN)
        )
        camera = Open3dCameraTrajectory.fromfile(input_json)
        poses = [
            np.array(p["extrinsic"], dtype=np.float32)
            .reshape(4, 4)
            .transpose()
            for p in camera.parameters
        ]
    else:
        raise ValueError(f"Unsupported camera pose file: {input_json}")

    poses = np.asarray(poses)
    assert poses.ndim == 3
    poses.shape[1] == 4
    poses.shape[2] == 4

    # apply scale
    poses[:, 0, 3] *= scale_x
    poses[:, 1, 3] *= scale_y
    poses[:, 2, 3] *= scale_z

    # apply move
    poses[:, 0, 3] += offset_x
    poses[:, 1, 3] += offset_y
    poses[:, 2, 3] += offset_z

    # apply rotation
    rot_vec = np.asarray([rotation_x, rotation_y, rotation_z]) / 180.0 * np.pi
    rotation = [
        R.from_matrix(pose[:3, :3]).as_euler("xyz") + rot_vec for pose in poses
    ]
    for pose, rot in zip(poses, rotation):
        pose[:3, :3] = R.from_euler("xyz", rot).as_matrix()

    if NerfCamera.is_valid(input_json):
        for frame, pose in zip(camera.frames, poses):
            frame["transform_matrix"] = pose.tolist()
    elif Open3dCameraTrajectory.is_valid(input_json):
        for parameter, pose in zip(camera.parameters, poses):
            parameter["extrinsic"] = pose.transpose().flatten().tolist()
    else:
        pass
    camera.dump(output)
    typer.echo(typer.style(f"save to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
