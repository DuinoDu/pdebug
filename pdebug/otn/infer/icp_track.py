"""

Depends:
    pyrealsense2
    cutoop
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

from pdebug.data_types import Camera, PointcloudTensor, Sam6dResult
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.ddd import generate_pcd_from_depth, transform_points
from pdebug.utils.env import RERUN_INSTALLED, TORCH_INSTALLED
from pdebug.utils.fileio import do_system
from pdebug.visp import rerun

import cv2
import numpy as np
import tqdm
import typer
from scipy.spatial.transform import Rotation as sstR

if RERUN_INSTALLED:
    import rerun as rr

if TORCH_INSTALLED:
    import torch

try:
    from cutoop.bbox import create_3d_bbox
    from cutoop.transform import transform_coordinates_3d
except ModuleNotFoundError:
    pass


def sam6d_camera_to_genpose2_meta(camera_file):
    """Convert sam6d camera.json to GenPose2 000000_meta.json format."""
    # Read camera.json
    with open(camera_file, "r") as f:
        camera_data = json.load(f)

    # Extract camera matrix and transform
    cam_K = np.array(camera_data["cam_K"]).reshape(3, 3)
    depth_scale = camera_data.get("depth_scale", 1.0)

    # Extract intrinsic parameters
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    # Create GenPose2 meta format
    meta_data = {
        "objects": {},
        "camera": {
            "quaternion": [1, 0, 0, 0],
            "translation": [0, 0, 0],
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(cx * 2),
                "height": int(cy * 2),
            },
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": "",
        },
        "scene_dataset": "real",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": "converted from sam6d camera.json",
        "runtime_seed": -1,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }
    return meta_data


def vggt_camera_to_genpose2_meta(camera_file, image_shape):
    """Convert sam6d camera.json to GenPose2 000000_meta.json format."""
    # Read camera.json
    with open(camera_file, "r") as f:
        camera_data = json.load(f)

    # Extract camera matrix and transform
    cam_K = np.asarray(camera_data)
    depth_scale = 1.0

    # Extract intrinsic parameters
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    # Create GenPose2 meta format
    meta_data = {
        "objects": {},
        "camera": {
            "quaternion": [1, 0, 0, 0],
            "translation": [0, 0, 0],
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(image_shape[1]),
                "height": int(image_shape[0]),
            },
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": "",
        },
        "scene_dataset": "real",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": "converted from vggt camera.json",
        "runtime_seed": -1,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
    }
    return meta_data


def generate_cam_bbox3d(
    sRT_4x4,
    bbox_side_len,
    scale=1.0,
    rotation=0.0,
    rotation_axis=[0.0, 0.0, 1.0],
    translation=[0.0, 0.0, 0.0],
):
    obj_bbox3d = create_3d_bbox(bbox_side_len)
    obj_bbox3d = transform_points(
        obj_bbox3d.T, scale, rotation, rotation_axis, translation
    ).T
    cam_bbox3d = transform_coordinates_3d(obj_bbox3d, sRT_4x4)
    return cam_bbox3d


def rotate_bbox3d(
    cam_bbox3d,
    sRT_4x4,
    scale=1.0,
    rotation=0.0,
    rotation_axis=[0.0, 0.0, 1.0],
    translation=[0.0, 0.0, 0.0],
):
    # cam_bbox3d: [3, N]
    obj_bbox3d = transform_coordinates_3d(cam_bbox3d, np.linalg.inv(sRT_4x4))
    obj_bbox3d = transform_points(
        obj_bbox3d.T, scale, rotation, rotation_axis, translation
    ).T
    cam_bbox3d = transform_coordinates_3d(obj_bbox3d, sRT_4x4)
    return cam_bbox3d


def computeRtFromTwoPoints(source_points, target_points):
    assert source_points.shape == target_points.shape == (8, 3)

    # 1. 计算质心
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)

    # 2. 去质心
    src_centered = source_points - centroid_src
    tgt_centered = target_points - centroid_tgt

    # 3. 计算协方差矩阵
    H = np.dot(src_centered.T, tgt_centered)

    # 4. SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 5. 计算旋转矩阵
    R = np.dot(Vt.T, U.T)

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 6. 计算平移向量
    t = centroid_tgt - np.dot(R, centroid_src)

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


@otn_manager.NODE.register(name="icp_track")
def main(
    rgb_path: str = None,
    depth_path: str = None,
    mask_path: str = None,
    sam6d_mask_path: str = None,
    sam6d_camera_path: str = None,
    vggt_camera_path: str = None,
    meta_path: str = None,
    repo: str = None,
    output: str = typer.Option(
        "./genpose2_output", help="Output directory for results"
    ),
    save_to_sam6d: bool = typer.Option(
        False, help="Save results to sam6d json"
    ),
    vis_output: str = typer.Option(
        None, help="Visualized output directory for results"
    ),
    vis_fps: int = typer.Option(10, help="Visualized video fps"),
    #####
    score_model: str = typer.Option(
        "results/ckpts/ScoreNet/scorenet.pth", help="Path to score model"
    ),
    energy_model: str = typer.Option(
        "results/ckpts/EnergyNet/energynet.pth", help="Path to energy model"
    ),
    scale_model: str = typer.Option(
        "results/ckpts/ScaleNet/scalenet.pth", help="Path to scale model"
    ),
    topk: int = typer.Option(-1, help="topk images"),
    det_score_topk: int = typer.Option(1, help="topk scores in detections"),
    skip_if_depth_not_exists: bool = typer.Option(
        False,
        help="skip current image if depth not found, used when input depth files are less than rgb files",
    ),
    vis_rerun: bool = False,
    rr_ip: str = None,
    erode_on_mask: bool = False,
    save_first_pointcloud: bool = False,
    load_first_annotation: str = None,
):
    """GenPose2 model inference pipeline.

    Args:
        repo: Path to GenPose2 repository
        output: Output directory for results
        score_model: Path to score model checkpoint
        energy_model: Path to energy model checkpoint
        scale_model: Path to scale model checkpoint
    """
    (
        output,
        save_to_sam6d,
        vis_output,
        vis_fps,
        score_model,
        energy_model,
        scale_model,
        topk,
        det_score_topk,
        skip_if_depth_not_exists,
    ) = otn_manager.check_optional(
        output,
        save_to_sam6d,
        vis_output,
        vis_fps,
        score_model,
        energy_model,
        scale_model,
        topk,
        det_score_topk,
        skip_if_depth_not_exists,
    )

    # Expand paths
    repo = Path(repo).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if save_to_sam6d:
        assert sam6d_mask_path
        assert sam6d_camera_path
        sam6d_mask_path = Path(sam6d_mask_path).expanduser().resolve()
    else:
        output.mkdir(parents=True, exist_ok=True)

    if vis_output:
        vis_output = Path(vis_output).expanduser().resolve()
        vis_output.mkdir(parents=True, exist_ok=True)
        vis_writer = Output(
            vis_output / "obj_pose.mp4", name="video_ffmpeg", fps=vis_fps
        ).get_writer()

    # Check if repo exists
    if not repo.exists():
        raise RuntimeError(f"GenPose2 repository not found at {repo}")

    # Add repo to sys.path and imports
    sys.path.insert(0, str(repo))
    from cutoop.data_loader import Dataset
    from runners.infer import GenPose2, InferDataset, visualize_pose

    # Model paths
    score_model_path = repo / score_model
    energy_model_path = repo / energy_model
    scale_model_path = repo / scale_model

    # Check model files exist
    for model_path in [score_model_path, energy_model_path, scale_model_path]:
        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")

    typer.echo(
        typer.style(f"Loading GenPose2 from {repo}", fg=typer.colors.GREEN)
    )

    # Initialize GenPose2
    genpose2 = GenPose2(
        score_model_path=str(score_model_path),
        energy_model_path=str(energy_model_path),
        scale_model_path=str(scale_model_path),
    )

    if Path(rgb_path).is_file():
        reader = Input(rgb_path, name="video", topk=topk).get_reader()
    else:
        reader = Input(rgb_path, name="imgdir", topk=topk).get_reader()
    assert len(reader), "Found 0 rgb files"

    depth_files = (
        Input(depth_path, name="imgdir", topk=topk).get_reader().imgfiles
    )
    assert len(depth_files), "Found 0 depth files"
    same_depth_files = len(depth_files) == len(reader)
    if skip_if_depth_not_exists:
        depth_files_map = {Path(f).stem: f for f in depth_files}

    if mask_path:
        mask_files = Input(mask_path, name="imgdir").get_reader().imgfiles
    typer.echo(
        typer.style(f"Found {len(reader)} color images", fg=typer.colors.GREEN)
    )

    if vis_rerun:
        assert rr_ip, "please provide rerun service ip address by --rr-ip."
        rerun.init_rerun(Path(rgb_path).stem, rr_ip)

    if save_first_pointcloud:
        (output / "pointcloud").mkdir(parents=True, exist_ok=True)

    prev_pose = None
    prev_length = None
    prev_pcd = None
    prev_bbox3d = None

    # Process each image
    t = tqdm.tqdm(total=len(reader))
    for index, color_image in enumerate(reader):
        t.update()

        if same_depth_files:
            depth_image = depth_files[index]
        elif skip_if_depth_not_exists:
            target_depth_filename = f"{index:06d}"
            if target_depth_filename not in depth_files_map:
                continue
            depth_image = depth_files_map[target_depth_filename]
        else:
            __import__("ipdb").set_trace()
            raise NotImplementedError

        base_name = Path(reader.filename).stem
        if mask_path:
            mask_img = cv2.imread(mask_files[index], -1)
        elif sam6d_mask_path:
            sam6d_result_path = sam6d_mask_path / base_name / "sam6d_results"
            sam6d_result = Sam6dResult(
                sam6d_result_path, det_score_topk=det_score_topk
            )
            # merge masks to one channel
            mask_img = None
            for idx, mask in enumerate(sam6d_result.get_masks("ism")):
                # mask is 0-1
                if idx == 0:
                    mask_img = mask
                else:
                    mask_img[mask] = idx + 1
        else:
            raise ValueError(
                "Please provide mask by --mask-path or --sam6d-mask-path"
            )

        if meta_path:
            meta_data = Dataset.load_meta(meta_path)
        elif sam6d_camera_path:
            meta_data = sam6d_camera_to_genpose2_meta(sam6d_camera_path)
        elif vggt_camera_path:
            meta_data = vggt_camera_to_genpose2_meta(
                vggt_camera_path, color_image.shape
            )
        else:
            raise ValueError(
                "Please provide meta(camera) by --meta-path or --sam6d-camera-path"
            )

        # Load data
        color_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(depth_image, -1).astype(np.float32)
        depth_img = depth_img / 1000.0  # Default millimeters to meters

        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]
        if mask_img.max() == 255:
            mask_img = mask_img // 255

        # enrode on mask_img
        if erode_on_mask:
            kernel = np.ones((10, 10), np.uint8)
            mask_img = cv2.erode(mask_img, kernel, iterations=1)

        if vis_rerun:
            intrs = meta_data["camera"]["intrinsics"]
            camera = Camera(
                np.eye(4), [intrs["fx"], intrs["fy"], intrs["cx"], intrs["cy"]]
            )
            depth_img_rr = depth_img.copy()
            depth_img_rr[~mask_img.astype(bool)] = 0
            vc = rr.ViewCoordinates.RFU
            rerun.rr_rgbd(
                color_image,
                depth_img_rr,
                camera,
                timestamp=index,
                view_coordinate=vc,
            )

        from pdebug.otn.data import icp_node

        intrs = meta_data["camera"]["intrinsics"]
        camera = Camera(
            np.eye(4), [intrs["fx"], intrs["fy"], intrs["cx"], intrs["cy"]]
        )
        depth_img_obj = depth_img.copy()
        depth_img_obj[~mask_img.astype(bool)] = 0

        T_open3d_genpose2 = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        T_genpose2_open3d = np.linalg.inv(T_open3d_genpose2)

        cur_pcd = generate_pcd_from_depth(depth_img_obj, camera)

        if prev_pcd is None:
            T_cur_prev = None
            prev_pcd = cur_pcd
            if save_first_pointcloud:
                cur_pcd.to_ply_ascii(
                    str(output / "pointcloud" / f"{index:06d}.ply")
                )

        else:
            T_cur_prev, fitness, rmse = icp_node.run_icp_registration(
                prev_pcd,
                cur_pcd,
                distance_threshold=0.005,
                method="point_to_plane",
            )
            pred_pcd = icp_node.apply_transformation(prev_pcd, T_cur_prev)
            # # Use pred_pcd from icp as prev_pcd, it contains accumulated error introduced by icp
            # prev_pcd = pred_pcd
            # Use cur_pcd as prev_pcd
            prev_pcd = cur_pcd

            if vis_rerun:
                # # Visualize icp result
                # pcd_source = prev_pcd.copy().fill_color([255, 0, 0]).transform(T_genpose2_open3d)
                # pcd_target = cur_pcd.copy().fill_color([0, 255, 0]).transform(T_genpose2_open3d)
                # pcd_transfomed = pred_pcd.copy().fill_color([0, 0, 255]).transform(T_genpose2_open3d)
                # rr.log("world/icp_source", rr.Points3D(pcd_source.data, colors=pcd_source.color))
                # rr.log("world/icp_transformed", rr.Points3D(pcd_transfomed.data, colors=pcd_transfomed.color))
                # rr.log("world/icp_target", rr.Points3D(pcd_target.data, colors=pcd_target.color))
                # pcd_transfomed.color = None

                if prev_bbox3d is not None:
                    trans_bbox3d = (
                        icp_node.apply_transformation(
                            PointcloudTensor(prev_bbox3d), T_cur_prev
                        )
                        .copy()
                        .data
                    )
                    prev_bbox3d = trans_bbox3d
                    trans_bbox3d_rr = (
                        PointcloudTensor(trans_bbox3d)
                        .transform(T_genpose2_open3d)
                        .copy()
                        .data
                    )
                    colors = [0x00FFFFFF for _ in range(prev_bbox3d.shape[0])]
                    radii = [0.01 for _ in range(prev_bbox3d.shape[0])]
                    rr.log(
                        "world/bbox3d_trans",
                        rr.Points3D(
                            trans_bbox3d_rr, colors=colors, radii=radii
                        ),
                    )

        if index == 0:
            with open(load_first_annotation, "r") as fid:
                annotation = json.load(fid)

            vx = np.asarray([v["x"] for v in annotation["vertices"]])[:, None]
            vy = np.asarray([v["y"] for v in annotation["vertices"]])[:, None]
            vz = np.asarray([v["z"] for v in annotation["vertices"]])[:, None]
            vertices = np.concatenate((vx, vy, vz), axis=1)

            anno_scale = np.asarray(
                [
                    annotation["scale"]["width"],
                    annotation["scale"]["height"],
                    annotation["scale"]["depth"],
                ]
            )
            obj_bbox3d = create_3d_bbox(anno_scale).T
            pose = computeRtFromTwoPoints(obj_bbox3d, vertices)
            # transformed = PointcloudTensor(obj_bbox3d).transform(pose)
            length = [torch.tensor(anno_scale)[None, :]]
            pose = [torch.tensor(pose)[None, :, :]]
        else:
            pose = [
                torch.tensor(T_cur_prev, dtype=prev_pose[0].dtype)
                @ prev_pose[0]
            ]
            length = prev_length

        if vis_output:
            # Visualize results
            data_dict = {
                "color": color_img,
                "depth": depth_img,
                "mask": mask_img,
                "meta": meta_data,
            }
            data = InferDataset(
                data_dict,
                img_size=genpose2.cfg.img_size,
                device=genpose2.cfg.device,
                n_pts=genpose2.cfg.num_points,
            )
            color_image_w_pose = visualize_pose(
                data,
                pose,
                length,
            )
            color_image_w_pose = cv2.putText(
                color_image_w_pose,
                str(index),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            vis_writer.write_frame(color_image_w_pose)
            if vis_rerun:
                rr.log(
                    "world/genpose2",
                    rr.Image(color_image_w_pose, color_model="BGR").compress(
                        jpeg_quality=70
                    ),
                )

            # Calcute boxes3d from pose and length
            bbox_side_len = length[0][0].cpu().numpy()
            sRT_4x4 = pose[0][0].cpu().numpy()
            cur_bbox3d = generate_cam_bbox3d(sRT_4x4, bbox_side_len).T
            if prev_bbox3d is None:
                # insert into prev_pcd
                prev_pcd.data = np.concatenate(
                    (prev_pcd.data, cur_bbox3d), axis=0
                )
                prev_bbox3d = cur_bbox3d

        # Save pose results
        pose_np = pose[0].cpu().numpy()
        length_np = length[0].cpu().numpy()

        result = {
            "image": reader.filename,
            "pose": pose_np.tolist(),
            "length": length_np.tolist(),
            "index": index,
        }
        result_file = output / f"{base_name}_pose.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Update previous pose for tracking
        prev_pose = pose
        prev_length = length

    if vis_output:
        vis_writer.save()

    if vis_rerun:
        print("Waiting for rerun to finish ...")
        time.sleep(3)
        print("Done.")


if __name__ == "__main__":
    typer.run(main)
