import glob
import json
import math
import os
import random
from pathlib import Path

from pdebug.data_types import PointcloudTensor
from pdebug.geometry import Vector3d, WorldX, WorldY, WorldZ
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.piata.coco.utils import binary_mask_to_bbox, binary_mask_to_rle
from pdebug.utils.ddd import get_3D_corners, load_points_3d_from_cad
from pdebug.utils.env import TRIMESH_INSTALLED
from pdebug.utils.fileio import do_system

import cv2
import numpy as np
import pycocotools.mask as cocomask
import tqdm
import typer
import yaml

if TRIMESH_INSTALLED:
    import trimesh
else:
    trimesh = None


@otn_manager.NODE.register(name="sam6d")
def main(
    rgb_path: str,
    depth_path: str,
    cad_path: str,
    camera_path: str,
    repo: str,
    output: str = None,
    cache: bool = True,
    skip_pose: bool = False,
    seg_first_frame: bool = False,
):
    """Sam6d auto annotation pipeline.

    Args:
        path: input imgdir
        cad_path: CAD file path
    """
    if not output:
        output = "tmp_sam6d_output"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    BLENDER_INSTALL_PATH = os.path.expanduser("~/opt")
    repo = os.path.expanduser(repo)
    render_file = Path(repo) / "SAM-6D/Render/render_custom_templates.py"
    if not render_file.exists():
        raise RuntimeError(f"{render_file} not exists.")

    render_output = output / "templates"
    if cache and render_output.exists():
        typer.echo(
            typer.style(f"{render_output} exists, skip", fg=typer.colors.GREEN)
        )
    else:
        render_cmd = f"blenderproc run {render_file} --output_dir {output} --cad_path {cad_path} --blender-install-path {BLENDER_INSTALL_PATH}"
        ret = do_system(render_cmd, with_return=True, trials=10)
        if not ret:
            raise RuntimeError(f"{render_cmd} failed")

    assert os.path.exists(cad_path), f"CAD {cad_path} not found."
    assert os.path.exists(camera_path), f"Camera {camera_path} not found."
    rgb_files = Input(rgb_path, name="imgdir").get_reader().imgfiles
    assert len(rgb_files), "Found 0 rgb files"
    depth_files = Input(depth_path, name="imgdir").get_reader().imgfiles
    assert len(depth_files), "Found 0 depth files"

    repo = os.path.expanduser(repo)
    seg_root = Path(repo) / "SAM-6D" / "Instance_Segmentation_Model"
    pose_root = Path(repo) / "SAM-6D" / "Pose_Estimation_Model"
    SEGMENTOR_MODEL = "sam"  # "fastsam"

    if seg_first_frame:
        rgb_path = (
            Input(rgb_path, name="imgdir").get_reader().imgfiles[0] + "*"
        )
        depth_path = (
            Input(depth_path, name="imgdir").get_reader().imgfiles[0] + "*"
        )
        skip_pose = True

    seg_cmd = (
        f"cd {seg_root}; python3 run_inference_custom.py --segmentor_model {SEGMENTOR_MODEL} "
        f"--output_dir {output} --cad_path {cad_path} --rgb_path '{rgb_path}' --depth_path '{depth_path}' "
        f"--template_dir {render_output.resolve()} --cam_path {camera_path}"
    )
    ret = do_system(seg_cmd, with_return=True, trials=20)
    if not ret:
        raise RuntimeError(f"{seg_cmd} failed")

    if not skip_pose:
        pose_cmd = (
            f"cd {pose_root}; python run_inference_custom.py "
            f"--output_dir {output} --cad_path {cad_path} --rgb_path '{rgb_path}' --depth_path '{depth_path}' "
            f"--template_dir {render_output.resolve()} --cam_path {camera_path} --det_score_topk 2"
        )
        ret = do_system(pose_cmd, with_return=True, trials=20)
        if not ret:
            raise RuntimeError(f"{pose_cmd} failed")


def project_3d_to_2d(points_3d, R, t, K):
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points in object frame
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        K: (3, 3) camera intrinsic matrix

    Returns:
        points_2d: (N, 2) array of 2D image coordinates
    """
    # Transform 3D points to camera coordinate system
    points_cam = (R @ points_3d.T).T + t

    # Project to 2D
    points_homogeneous = K @ points_cam.T
    points_2d = points_homogeneous[:2, :] / points_homogeneous[2:3, :]

    return points_2d.T


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def align_bbox_to_world(bbox_3d, R, t, T_world_cam, object_x_idx):
    """Align bbox to world axis.

    Args:
        bbox_3d: [9, 3]
        R: rotation matrix, [3, 3]
        t: translation
        T_world_cam: T from camera to world, [4, 4]
        object_x_idx: x direction vec of object
    """
    # obj to camera
    points_cam = (R @ bbox_3d.T).T + t

    # camera to world
    points_cam_hom = np.column_stack(
        [points_cam, np.ones(points_cam.shape[0])]
    )
    points_world_hom = (T_world_cam @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3]

    # angle between bbox_3d X-axis with world X-axis
    target_axis = WorldX
    obj_x_vec = Vector3d(
        *(
            (
                points_world[object_x_idx[0]] - points_world[object_x_idx[1]]
            ).tolist()
        )
    )
    angle_with_x = obj_x_vec.angle_to(target_axis)

    # compute rotation axis and rotation matrix
    k = np.cross(obj_x_vec.numpy(), target_axis.numpy())
    if np.linalg.norm(k) < 1e-6:
        if np.dot(v, target_axis.numpy()) > 0:
            rot_mat = np.eye(3)
        else:
            rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        k = k / np.linalg.norm(k)
        # Rodrigues
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot_mat = (
            np.eye(3)
            + np.sin(angle_with_x) * K
            + (1 - np.cos(angle_with_x)) * np.dot(K, K)
        )
    R = R @ rot_mat

    # PointcloudTensor(points_world).to_ply("aa.ply")
    # points_cam = (R @ bbox_3d.T).T + t
    # points_cam_hom = np.column_stack([points_cam, np.ones(points_cam.shape[0])])
    # points_world_hom = (T_world_cam @ points_cam_hom.T).T
    # points_world = points_world_hom[:, :3]
    # PointcloudTensor(points_world).to_ply("ab.ply")
    return R


@otn_manager.NODE.register(name="sam6d-to-linemod")
def to_linemod(
    path: str,
    rgb: str,
    cad: str,
    camera: str,
    category: str,
    output: str = "tmp_linemod",
    max_train_num: int = 200,
    #### cleaning data parameters
    seg_score_thresh: float = 0.5,
    max_det_num: int = -1,
    background_imgdir: str = None,
    align_to_world: bool = False,
):
    """
    Convert sam6d output to linemod format.

    Args:
        path: Path to sam6d output.
        rgb: RGB image filename pattern or path relative to dataset directory
        cad: 3D CAD model filename for the object
        camera: Camera intrinsic parameters filename
        category: Object category name
        output: Output directory for LineMOD format dataset (default: "tmp_linemod")
        seg_score_thresh: Minimum segmentation score threshold for valid masks (default: 0.5)
        max_det_num: Maximum detection instances from sam6d.
        max_train_num: Maximum number of training samples to generate (default: 200)
        background_imgdir: Background image folder, used by yolo-pose training.
        align_to_world: Align object pose (R) to world axis. Only for circular symmetry object,
                        or spherical symmetry object.

    Returns:
        None, creates LineMOD format dataset in specified output directory

    LINEMOD dataset format:
        --- output/{category}
                        |---JPEGImages
                               |--- 000000.jpg
                        |---mask
                               |--- 0000.png
                        |---labels
                               |--- 000000.txt
                        |---{category}.ply
                        |---linemod_camera.json
                        |---train.txt
                        |---test.txt
                        |---{category}.yaml         # yolopose config

    label txt format:
        - one line for one object, with 27 float
        - 0: class_id
        - 1:19: corner2d, x y * 9, normalize by w/h, first one is center point
        - 19:21: width, height
        - 22:27: camera intrinsic

        X   Y   Z
       ===========
       +1  +1  +1
       +1  +1  -1
       -1  +1  +1
       -1  +1  -1
       +1  -1  +1
       +1  -1  -1
       -1  -1  +1
       -1  -1  -1

    """
    output = os.path.join(output, category)
    os.makedirs(output, exist_ok=True)

    rgb_files = Input(rgb, name="imgdir").get_reader().imgfiles
    typer.style(f"Found {len(rgb_files)} rgb files", fg=typer.colors.GREEN)

    imgdir = os.path.join(output, "JPEGImages")
    maskdir = os.path.join(output, "mask")
    labeldir = os.path.join(output, "labels")

    # load bbox_3d from cad
    vertices = load_points_3d_from_cad(cad, dst_unit="millimeter", legacy=True)
    bbox_3d = get_3D_corners(vertices)

    # load K from camera
    with open(camera, "r") as f:
        camera_data = json.load(f)
    K = np.array(camera_data["cam_K"]).reshape(3, 3)
    if "T_world_cam" in camera_data:
        T_world_cam = np.array(camera_data["T_world_cam"]).reshape(4, 4)
    else:
        # T_world_cam = None
        # TODO: Remove hard code
        T_world_cam = [
            1.0,
            0.0,
            0.0,
            -0.03200000524520874,
            -0.0,
            -0.7999999737739569,
            0.5999999952316282,
            -0.45000002920627497,
            -0.0,
            -0.5999999952316282,
            -0.799999973773957,
            1.349999974370003,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        T_world_cam = np.array(T_world_cam).reshape(4, 4)

    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(maskdir, exist_ok=True)
    os.makedirs(labeldir, exist_ok=True)
    valid_rgb_files = []
    progress = tqdm.tqdm(total=len(rgb_files))
    for ind, rgb_file in enumerate(rgb_files):
        progress.update()
        # print(f"{ind} / {len(rgb_files)}")

        name, ext = os.path.splitext(rgb_file)
        pose_path = os.path.join(
            path, os.path.basename(name), "sam6d_results/detection_pem.json"
        )
        if not os.path.exists(pose_path):
            continue
        dets = []
        dets_ = json.load(
            open(pose_path)
        )  # keys: scene_id, image_id, category_id, bbox, score, segmentation
        # print([d["score"] for d in dets_])
        dets_ = sorted(dets_, key=lambda input: input["score"], reverse=True)
        if max_det_num > 0:
            dets_ = dets_[:max_det_num]
        for det in dets_:
            if det["score"] > seg_score_thresh:
                dets.append(det)

        if not dets:
            msg = (
                f"No valid dets found in {rgb_file}: \n"
                f"  dets['scores']: {[d['score'] for d in dets_]}\n"
                f"  seg_score_thresh: {seg_score_thresh}\n"
                f"  max_det_num: {max_det_num}"
            )
            typer.echo(typer.style(msg, fg=typer.colors.YELLOW))
            del dets_
            continue
        del dets_

        dst = f"{imgdir}/{ind:06d}{ext}"
        if not os.path.exists(dst):
            os.system(f"ln -s {rgb_file} {dst}")

        mask_ch3 = None
        labels = []
        for inst in dets:
            seg = inst["segmentation"]
            # mask
            h, w = seg["size"]
            if mask_ch3 is None:
                mask_ch3 = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                rle = cocomask.frPyObjects(seg, h, w)
            except:
                rle = seg
            mask = cocomask.decode(rle)
            # 0 or 255, channel=3
            mask_ch3[:, :, 0] = np.logical_or(mask_ch3[:, :, 0], mask)
            mask_ch3[:, :, 1] = np.logical_or(mask_ch3[:, :, 1], mask)
            mask_ch3[:, :, 2] = np.logical_or(mask_ch3[:, :, 2], mask)

            if (
                "genpose2_length" in inst
                and "R_genpose2" in inst
                and "t_genpose2" in inst
            ):
                # re-compute bbox_3d from genpose2_length
                from cutoop.bbox import create_3d_bbox
                from cutoop.transform import (
                    calculate_2d_projections,
                    transform_coordinates_3d,
                )

                bbox_3d = create_3d_bbox(inst["genpose2_length"]) / 1000
                bbox_3d = np.concatenate(
                    (bbox_3d.mean(1)[:, None], bbox_3d), axis=1
                )
                object_x_idx = [1, 2]
                R = np.array(inst["R_genpose2"])
                t = np.array(inst["t_genpose2"])
                sRT_4x4 = np.eye(4)
                sRT_4x4[:3, :3] = R
                sRT_4x4[:3, 3] = t / 1000  # mm to meter
                transformed_bbox_Nx3 = transform_coordinates_3d(
                    bbox_3d, sRT_4x4
                )
                bbox_2d = calculate_2d_projections(
                    transformed_bbox_Nx3, K
                ).astype(np.float32)

            else:
                object_x_idx = [1, 3]
                R = np.array(inst["R"])
                t = np.array(inst["t"])

                if align_to_world:
                    assert T_world_cam is not None
                    R = align_bbox_to_world(
                        bbox_3d, R, t, T_world_cam, object_x_idx
                    )

                bbox_2d = project_3d_to_2d(bbox_3d, R, t, K)

            # import sys
            # sys.path.append("/home/duino/code/github/SAM-6D/SAM-6D/Pose_Estimation_Model/utils")
            # from draw_utils import draw_detections
            # vis_bbox = draw_detections(
            #         cv2.imread(rgb_file),
            #         R[None, :], t[None, :], vertices, K[None, :], color=(255, 0, 0))
            # from pdebug.visp import draw
            # vis_bbox_2 = draw.points(cv2.imread(rgb_file), bbox_2d)
            # vis_bbox = np.concatenate((vis_bbox, vis_bbox_2), axis=1)
            # cv2.imwrite(f"vis_{ind}.png", vis_bbox)
            # __import__('ipdb').set_trace()
            # pass

            label = [0.0]  # 0:1
            bbox_2d = bbox_2d.astype(np.float32)
            bbox_2d[:, 0] /= w
            bbox_2d[:, 1] /= h
            label.extend(bbox_2d.flatten())  # 1:19
            bbox_w, bbox_h = inst["bbox"][2:4]
            label.extend([bbox_w / w, bbox_h / h])  # 19:21
            label.extend([K[0, 0], K[1, 1], w, h])  # 21:25
            label.extend([K[0, 2], K[1, 2], w, h])  # 25:29
            label = [f"{v:.6f}" for v in label]
            labels.append(" ".join(label))

        mask_ch3 *= 255
        savename = os.path.join(maskdir, f"{ind:04d}.png")
        cv2.imwrite(savename, mask_ch3)

        savename = os.path.join(labeldir, f"{ind:06d}.txt")
        with open(savename, "w") as fid:
            for l in labels:
                fid.write(l + "\n")
        valid_rgb_files.append(dst)

    # {category}.ply
    ply_file = os.path.join(output, f"{category}.ply")
    vertices = load_points_3d_from_cad(
        cad, n_points=-1, dst_unit="meter", legacy=True
    )
    PointcloudTensor(vertices).to_ply_ascii(ply_file)

    # linemod_camera.json
    linemod_camera = {
        "distortion": None,
        "intrinsic": K.tolist(),
    }
    savename = os.path.join(output, "linemod_camera.json")
    json.dump(linemod_camera, open(savename, "w"))

    # train.txt, test.txt
    train_txt = os.path.join(output, "train.txt")
    test_txt = os.path.join(output, "test.txt")
    filelist = [os.path.basename(f) for f in valid_rgb_files]
    random.shuffle(filelist)
    num_train = min(max_train_num, len(filelist) // 2)
    train_list = filelist[:num_train]
    test_list = filelist[num_train:]
    with open(train_txt, "w") as fid:
        for name in train_list:
            fid.write(name + "\n")
    with open(test_txt, "w") as fid:
        for name in test_list:
            fid.write(name + "\n")

    # {category}.yaml
    diameter = calc_pts_diameter(vertices)
    yolopose_cfg = {
        "train": train_txt,
        "val": test_txt,
        "test": test_txt,
        "mesh": ply_file,
        "names": [category],
        "diam": diameter,
        "nc": 1,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "u0": float(K[0, 2]),
        "v0": float(K[1, 2]),
    }
    if background_imgdir:
        yolopose_cfg["background_path"] = background_imgdir
    with open(os.path.join(output, f"{category}.yaml"), "w") as f:
        yaml.dump(yolopose_cfg, f)

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    typer.echo(
        typer.style(
            f"  rgb file num: {len(valid_rgb_files)}", fg=typer.colors.GREEN
        )
    )


@otn_manager.NODE.register(name="sam6d-to-sam2")
def to_sam2(input_path, mask_path):
    """Convert first frame seg json in sam6d result to sam2 prompt."""
    input_path = Path(input_path) / "sam6d_results" / "detection_ism.json"
    with open(input_path, "r") as fid:
        dets = json.load(fid)

    dets = [d for d in dets if d["score"] > 0.5]
    assert len(dets) == 1
    inst = dets[0]
    seg = inst["segmentation"]
    # mask
    h, w = seg["size"]
    try:
        rle = cocomask.frPyObjects(seg, h, w)
    except:
        rle = seg
    mask = cocomask.decode(rle)
    cv2.imwrite(mask_path, mask)
    print(f"saved to {mask_path}")
    return mask_path


@otn_manager.NODE.register(name="sam6d-from-sam2")
def from_sam2(input_path, output_path):
    """Convert sam2 result to seg json in sam6d."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    for mask_file in os.listdir(input_path):
        mask = cv2.imread(str(input_path / mask_file), 0)
        outdir = output_path / Path(mask_file).stem / "sam6d_results"
        outdir.mkdir(parents=True, exist_ok=True)

        jsonfile = outdir / "detection_ism.json"
        inst = {}
        inst["segmentation"] = binary_mask_to_rle(mask)
        inst["bbox"] = binary_mask_to_bbox(mask)
        inst["image_id"] = Path(mask_file).stem
        inst["score"] = 1.0
        inst["category_id"] = 1
        inst["scene_id"] = 0
        with open(jsonfile, "w") as fid:
            json.dump([inst], fid)

    print(f"saved to {output_path}")
    return output_path


if __name__ == "__main__":
    typer.run(main)
