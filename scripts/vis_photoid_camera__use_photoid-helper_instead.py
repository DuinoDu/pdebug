#!/usr/bin/env python
import glob
import random

from pdebug.data_types import PointcloudTensor, Pose
from pdebug.debug.nerf import vis_pose
from pdebug.piata import Input
from pdebug.utils.three_utils import threejs_pcd
from pdebug.visp.colormap import Colormap

import numpy as np
import typer


def main(
    data_recording_root: str,
    port: int = typer.Option(6006, help="http server port"),
    vr: bool = typer.Option(False, help="in vr mode"),
    scale: float = typer.Option(1.0, help="scene scale, used in vr mode"),
    sample: int = typer.Option(0, help="random sample camera pose"),
    topk: int = typer.Option(None, help="topk pose"),
    input_pcd: str = typer.Option(None, help="extra pointcloud file"),
):
    """Visualize photoid data-recording data camera pose."""
    all_poses, all_colors = [], []

    if "*" in data_recording_root:
        # grep pose txt files
        txtfiles = glob.glob(data_recording_root)
        txtfiles.sort()
        colormap = Colormap(len(txtfiles), hex_mode=True)

        for idx, txtfile in enumerate(txtfiles):
            pose_lines = open(txtfile, "r").readlines()
            assert len(pose_lines) == 1
            pose = [float(i) for i in pose_lines[0].split(" ")]
            assert len(pose) == 7
            x, y, z, rw, rx, ry, rz = pose
            camera_pose = Pose.fromRt([rx, ry, rz, rw], [x, y, z])
            all_poses.append(np.array([camera_pose.data]))
            all_colors.append(np.array([colormap[idx]]))
    else:
        reader = Input(
            data_recording_root,
            name="PhotoidRecording",
            check=True,
            quiet=True,
        ).get_reader()
        colormap = Colormap(len(reader), hex_mode=True)

        for idx, frame in enumerate(reader):
            frame.load_data(load_rgb=False, load_depth=False)
            if not frame.tof_camera:
                continue
            if sample > 0 and idx % sample != 0:
                continue

            pose = frame.tof_camera_pose
            all_poses.append(np.array([pose.data]))
            all_colors.append(np.array([colormap[idx]]))

    all_poses = np.concatenate(all_poses, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

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
        res_dir="/home/bingwen/tmp_vis_camera_pose",
    )


if __name__ == "__main__":
    typer.run(main)
