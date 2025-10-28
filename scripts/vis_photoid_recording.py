#!/usr/bin/env python
import copy
import os
from typing import Optional

from pdebug.data_types import Camera
from pdebug.piata import Input
from pdebug.utils.ddd import generate_pcd_from_depth, project_depth_to_rgb
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer

try:
    import pdebug.piata.handler.photoid_recording

    import photoid

    device_parameters = photoid.DeviceParameters()
    PHOTOID_INSTALLED = True
except ModuleNotFoundError as e:
    PHOTOID_INSTALLED = False


def vis_v1(frame, output, reader):
    """basic vis."""
    frame.load_data(load_rgb_right=True, load_depth_conf=True)

    left_rgb = frame.rgb
    right_rgb = frame.rgb_right
    left_rgb = cv2.putText(
        left_rgb,
        f"frame: {reader.index}",
        (100, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        4,
        (0, 0, 255),
        2,
    )

    vis_depth_width = 1000
    if frame.depth is not None:
        depth = frame.depth.astype(np.float32) * 0.001  # convert mm to m
        depth[depth > 5.0] = 0.0
        depth_conf = frame.depth_conf

        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth_conf = cv2.rotate(depth_conf, cv2.ROTATE_90_CLOCKWISE)

        vis_depth = draw.depth(depth, hist=True, vertical=False)
        vis_depth_conf = draw.depth(depth_conf, hist=True, vertical=False)

        scale = vis_depth.shape[1] / vis_depth_width
        vis_depth = cv2.resize(
            vis_depth,
            (int(vis_depth.shape[1] / scale), int(vis_depth.shape[0] / scale)),
        )
        vis_depth_conf = cv2.resize(
            vis_depth_conf,
            (
                int(vis_depth_conf.shape[1] / scale),
                int(vis_depth_conf.shape[0] / scale),
            ),
        )
        vis_padding = np.zeros(
            (
                left_rgb.shape[0] - vis_depth.shape[0] * 2,
                vis_depth.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        vis_right = np.concatenate(
            (vis_depth, vis_depth_conf, vis_padding), axis=0
        )
    else:
        vis_right = np.zeros(
            (left_rgb.shape[0], vis_depth_width, 3), dtype=np.uint8
        )

    vis = np.concatenate((left_rgb, right_rgb, vis_right), axis=1)
    cv2.imwrite(f"{output}/{reader.index:06d}.png", vis)


left_camera = None
tof_camera = None


def vis_v2_old(frame, output, reader, full, with_pcd=True):
    """vis using photoid"""
    assert PHOTOID_INSTALLED, "photoid not found, please install first."
    head_model_offset = device_parameters.head_model_offset
    T_B_I = device_parameters.T_B_I

    T_matrix_H_I = np.array(
        [
            [0, -1, 0, -head_model_offset[0]],
            [1, 0, 0, -head_model_offset[1]],
            [0, 0, 1, head_model_offset[2]],
            [0, 0, 0, 1],
        ]
    )
    T_H_I = photoid.SE3(T_matrix_H_I[:3, :3], T_matrix_H_I[:3, 3])

    global left_camera, tof_camera
    if left_camera is None:
        left_camera = photoid.Camera(reader.left_rgb_camera_file, T_B_I)
        left_camera.initUndistortRectifyMap(
            0.5
        )  # may cause ratio-mismatch problem
        tof_camera = photoid.Camera(reader.tof_camera_file, T_B_I)
    rgb_camera = left_camera

    frame.load_data(load_rgb_right=False, load_depth_conf=True)
    left_rgb = frame.rgb
    frame_index = frame.extras["index"]

    if (
        left_rgb.shape[0] != frame.left_rgb_camera.I.height
        or left_rgb.shape[1] != frame.left_rgb_camera.I.width
    ):
        left_rgb = cv2.resize(
            left_rgb,
            (frame.left_rgb_camera.I.width, frame.left_rgb_camera.I.height),
        )

    undistort_rgb = left_camera.undistortImage(left_rgb)
    # cv2.imwrite(f"{output}/{frame_index:06d}.png", undistort_rgb)
    # return

    rgb_pose = frame.extras[
        "data_batch"
    ].left_rgb_camera_image.tracking_info.pose.pose
    rgb_timestamp = frame.extras["data_batch"].left_rgb_camera_image.timestamp
    rgb_pose_se3 = photoid.SE3([float(p) for p in rgb_pose])

    vis_depth_width = 1000
    if frame.depth is not None:
        K_rgb = left_camera.camKnew
        K_tof = tof_camera.camK

        tof_pose = frame.extras[
            "data_batch"
        ].tof_depth_camera_image.tracking_info.pose.pose
        tof_timestamp = frame.extras[
            "data_batch"
        ].tof_depth_camera_image.timestamp
        tof_pose_se3 = photoid.SE3([float(p) for p in tof_pose])

        T_W_I_rgb = photoid.release_head_model(rgb_pose_se3, head_model_offset)
        T_W_I_tof = photoid.release_head_model(tof_pose_se3, head_model_offset)
        T_WH_H_rgb = photoid.ApplyHeadModel(T_W_I_rgb, head_model_offset)
        T_WH_H_tof = photoid.ApplyHeadModel(T_W_I_tof, head_model_offset)
        T_W_rgb = T_WH_H_rgb * T_H_I * rgb_camera.T_I_S
        T_W_tof = T_WH_H_tof * T_H_I * tof_camera.T_I_S
        T_rgb_tof = T_W_rgb.inverse() * T_W_tof

        depth = frame.depth.astype(np.float32) * 0.001  # convert mm to m
        depth[depth > 5.0] = 0.0
        depth_conf = frame.depth_conf

        if (
            depth.shape[0] != frame.tof_camera.I.height
            or depth.shape[1] != frame.tof_camera.I.width
        ):
            depth = cv2.resize(
                depth, (frame.tof_camera.I.width, frame.tof_camera.I.height)
            )
            depth_conf = cv2.resize(
                depth_conf,
                (frame.tof_camera.I.width, frame.tof_camera.I.height),
            )

        # # debug
        # # tof mode: vga to qvga
        # depth = cv2.resize(depth, (int(depth.shape[1]/2), int(depth.shape[0]/2)))
        # K_tof[:2, :3] /= 2

        depth_rot = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth_conf_rot = cv2.rotate(depth_conf, cv2.ROTATE_90_CLOCKWISE)
        vis_depth = draw.depth(depth_rot, hist=True, vertical=False)
        vis_depth_conf = draw.depth(depth_conf_rot, hist=True, vertical=False)

        scale1 = vis_depth.shape[1] / vis_depth_width
        scale2 = vis_depth_conf.shape[1] / vis_depth_width
        vis_depth = cv2.resize(
            vis_depth,
            (
                int(vis_depth.shape[1] / scale1),
                int(vis_depth.shape[0] / scale1),
            ),
        )
        vis_depth_conf = cv2.resize(
            vis_depth_conf,
            (
                int(vis_depth_conf.shape[1] / scale2),
                int(vis_depth_conf.shape[0] / scale2),
            ),
        )
        vis_padding = np.zeros(
            (
                left_rgb.shape[0] - vis_depth.shape[0] * 2,
                vis_depth.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        vis_right = np.concatenate(
            (vis_depth, vis_depth_conf, vis_padding), axis=0
        )

        depthmap = project_depth_to_rgb(
            depth, undistort_rgb, K_tof, K_rgb, T_rgb_tof.matrix(), stride=1
        )
        vis_depthmap = draw.depthpoint(depthmap, image=undistort_rgb)
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"frame: {frame_index}",
            (100, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"rgb: {rgb_timestamp}",
            (100, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"tof: {tof_timestamp}",
            (100, 300),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_right = np.concatenate((vis_right, vis_depthmap), axis=1)

        if with_pcd:
            camera = Camera(T_W_rgb.matrix(), K_rgb)
            pcd_bgr = undistort_rgb
            pcd_rgb = cv2.cvtColor(pcd_bgr, cv2.COLOR_BGR2RGB)
            pcd = generate_pcd_from_depth(
                depthmap, camera, rgb=pcd_rgb, coordinate_type="photoid"
            )
            pcd.to_ply(f"{output}/{frame_index:06d}.ply")
    else:
        vis_right = np.zeros(
            (left_rgb.shape[0], vis_depth_width, 3), dtype=np.uint8
        )
        vis_depthmap = copy.deepcopy(undistort_rgb)
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"frame: {frame_index}",
            (100, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"rgb: {rgb_timestamp}",
            (100, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_right = np.concatenate((vis_right, vis_depthmap), axis=1)

    if full:
        vis = np.concatenate((left_rgb, undistort_rgb, vis_right), axis=1)
    else:
        vis = vis_depthmap

    cv2.imwrite(f"{output}/{frame_index:06d}.png", vis)


def vis_v3(frame, output, reader, full, with_pcd=True):
    """vis using photoid"""
    assert PHOTOID_INSTALLED, "photoid not found, please install first."
    frame.load_data(load_rgb_right=False, load_depth_conf=True)
    frame_index = frame.extras["index"]

    rgb_timestamp = frame.extras["data_batch"].left_rgb_camera_image.timestamp
    reader.get_undistort_rgb(frame)
    undistort_rgb = frame.undistort_rgb
    left_rgb = frame.rgb
    assert left_rgb.shape == undistort_rgb.shape

    vis_depth_width = 1000
    if frame.depth is not None:
        tof_timestamp = frame.extras[
            "data_batch"
        ].tof_depth_camera_image.timestamp
        reader.get_depthmap_raw(frame)
        depth, depth_conf = frame.depth, frame.depth_conf
        if depth.max() <= 0:
            print("bad depth file, skip")
            return

        # # debug
        # # tof mode: vga to qvga
        # depth = cv2.resize(depth, (int(depth.shape[1]/2), int(depth.shape[0]/2)))
        # K_tof[:2, :3] /= 2

        depth_rot = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        depth_conf_rot = cv2.rotate(depth_conf, cv2.ROTATE_90_CLOCKWISE)
        vis_depth = draw.depth(depth_rot, hist=True, vertical=False)
        vis_depth_conf = draw.depth(depth_conf_rot, hist=True, vertical=False)

        scale1 = vis_depth.shape[1] / vis_depth_width
        scale2 = vis_depth_conf.shape[1] / vis_depth_width
        vis_depth = cv2.resize(
            vis_depth,
            (
                int(vis_depth.shape[1] / scale1),
                int(vis_depth.shape[0] / scale1),
            ),
        )
        vis_depth_conf = cv2.resize(
            vis_depth_conf,
            (
                int(vis_depth_conf.shape[1] / scale2),
                int(vis_depth_conf.shape[0] / scale2),
            ),
        )
        vis_padding = np.zeros(
            (
                left_rgb.shape[0] - vis_depth.shape[0] * 2,
                vis_depth.shape[1],
                3,
            ),
            dtype=np.uint8,
        )
        vis_right = np.concatenate(
            (vis_depth, vis_depth_conf, vis_padding), axis=0
        )

        reader.get_depthmap_aligned_to_rgb(frame, with_pcd=with_pcd)
        depthmap, pcd = frame.depth_aligned_to_rgb, frame.pcd
        vis_depthmap = draw.depthpoint(depthmap, image=undistort_rgb)
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"frame: {frame_index}",
            (100, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"rgb: {rgb_timestamp}",
            (100, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"tof: {tof_timestamp}",
            (100, 300),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_right = np.concatenate((vis_right, vis_depthmap), axis=1)

        if with_pcd:
            pcd.to_ply(f"{output}/{frame_index:06d}.ply")
    else:
        vis_right = np.zeros(
            (left_rgb.shape[0], vis_depth_width, 3), dtype=np.uint8
        )
        vis_depthmap = copy.deepcopy(undistort_rgb)
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"frame: {frame_index}",
            (100, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_depthmap = cv2.putText(
            vis_depthmap,
            f"rgb: {rgb_timestamp}",
            (100, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            4,
            (0, 0, 255),
            2,
        )
        vis_right = np.concatenate((vis_right, vis_depthmap), axis=1)

    if full:
        vis = np.concatenate((left_rgb, undistort_rgb, vis_right), axis=1)
    else:
        vis = vis_depthmap

    cv2.imwrite(f"{output}/{frame_index:06d}.png", vis)


# @task(name="my-tool")
def main(
    path: str,
    check_with_depth: Optional[bool] = typer.Option(
        False, help="only rgb with depth"
    ),
    v1: Optional[bool] = typer.Option(False, help="force using v1"),
    full: Optional[bool] = typer.Option(
        False, help="keep raw and undistort image"
    ),
    topk: Optional[int] = typer.Option(-1, help="topk images"),
    start: Optional[int] = typer.Option(0, help="start frame index"),
    end: Optional[int] = typer.Option(-1, help="end frame index"),
    output: Optional[str] = typer.Option("vis_photoid", help="output name"),
    with_pcd: Optional[bool] = typer.Option(
        True, help="skip pcd visualization"
    ),
):
    """Visualize photoid recording data."""
    # typer.echo(typer.style(f"hello, tool", fg=typer.colors.GREEN))
    assert start >= 0

    os.makedirs(output, exist_ok=True)

    reader = Input(
        path,
        name="PhotoidRecording",
        check=check_with_depth,
        quiet=check_with_depth,
    ).get_reader()
    if len(reader) == 0:
        print(f"Found 0 images in {path}.")
        return

    t = tqdm.tqdm(total=len(reader), desc="process frame")
    for frame in reader:
        t.update()

        if reader.index < start:
            continue
        if end > 0 and reader.index >= end:
            continue
        if topk > 0 and reader.index == topk:
            break
        if PHOTOID_INSTALLED and (not v1):
            # vis_v2_old(frame, output, reader, full, with_pcd)
            vis_v3(frame, output, reader, full, with_pcd)
        else:
            vis_v1(frame, output, reader)


if __name__ == "__main__":
    typer.run(main)
