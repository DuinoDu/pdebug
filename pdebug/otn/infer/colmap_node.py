import json
import math
import os
import shutil
import sys
from shutil import which
from typing import Optional

from pdebug.data_types import CameraIntrinsic, NerfCamera, PointcloudTensor
from pdebug.otn import manager as otn_manager
from pdebug.utils.env import OPEN3D_INSTALLED
from pdebug.utils.fileio import do_system

import cv2
import numpy as np
import typer

if OPEN3D_INSTALLED:
    import open3d as o3d


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


@otn_manager.NODE.register(name="colmap_to_nerf")
def colmap_to_nerf(
    imgdir: str = "images",
    textdir: str = "colmap_text",
    aabb_scale: int = 16,
    skip_early: int = 0,
    align_first_pose_to_origin: bool = False,
    output: str = "nerf_result",
):
    """Convert colmap pose result to nerf camera format."""
    AABB_SCALE = int(aabb_scale)
    SKIP_EARLY = int(skip_early)
    os.makedirs(output, exist_ok=True)

    with open(os.path.join(textdir, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi
    print(
        f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} "
    )

    with open(os.path.join(textdir, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(
                    " "
                )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(imgdir, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(imgdir)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])

                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

                c2w = np.linalg.inv(m).tolist()

                frame = {
                    "file_path": name,
                    "sharpness": b,
                    "transform_matrix": c2w,
                }
                out["frames"].append(frame)

    nerf_camera = NerfCamera(out)
    if align_first_pose_to_origin:
        t_mat = nerf_camera.align_first_pose_to_origin()

    print(f"{len(nerf_camera)} frames")
    print(f"writing {output}")
    output_file = os.path.join(output, "transforms.json")
    nerf_camera.dump(output_file)

    points_txt = os.path.join(textdir, "points3D.txt")
    if os.path.exists(points_txt):
        if OPEN3D_INSTALLED:
            pcd = PointcloudTensor.from_colmap_point3d(points_txt).to_open3d()
            if align_first_pose_to_origin:
                pcd.transform(t_mat)
            output_file = os.path.join(output, "points.pcd")
            o3d.io.write_point_cloud(output_file, pcd, print_progress=True)
        else:
            print("open3d not found, skip save to o3d pcd file.")

    print(f"save to {output}")
    return output


@otn_manager.NODE.register(name="colmap_to_open3d")
def colmap_to_open3d(
    path: str,
    ext: str = ".pcd",
    output: str = None,
    cache: bool = False,
):
    """Convert Point3d in colmap to open3d pcd."""
    if path.endswith(".txt"):
        txtfile = path
    else:
        txtfile = os.path.join(path, "sparse/0/points3D.txt")
    assert os.path.exists(txtfile)

    if not output:
        output = txtfile[:-4] + ext

    if os.path.exists(output) and cache:
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output

    pcd = PointcloudTensor.from_colmap_point3d(txtfile)

    if output.endswith(".pcd"):
        pcd.to_open3d(output)
    elif output.endswith(".ply"):
        pcd.to_ply(output)
    print(f"save to {output}")
    return output


@otn_manager.NODE.register(name="imgdir_to_colmap")
def imgdir_to_colmap(
    imgdir: str,
    matcher: str = "sequential",  # choices=["exhaustive","sequential","spatial","transitive","vocab_tree"]
    camera_model: str = "OPENCV",  # choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL","OPENCV"]
    camera_params: str = "",  # help="intrinsic parameters, depending on the chosen model.  Format: fx,fy,cx,cy,dist",
    output: str = None,
    cache: bool = False,
    do_undistort_image: bool = True,
    skip_sparse: bool = False,
    do_dense: bool = False,
):
    """Imgdir to pose and sparse pcd, using colmap."""

    if not which("colmap"):
        typer.echo(
            typer.style(
                "colmap not found, forget to install colmap?",
                fg=typer.colors.RED,
            )
        )
        sys.exit()

    if not output:
        output = os.path.abspath(imgdir) + "_colmap"
    if os.path.exists(output):
        if cache:
            typer.echo(
                typer.style(f"Found {output}, skip", fg=typer.colors.WHITE)
            )
            return output
        if not skip_sparse:
            shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)
    db = os.path.join(output, "colmap.db")

    # step1: extract sift
    do_system(
        f'colmap feature_extractor --ImageReader.camera_model {camera_model} --ImageReader.camera_params "{camera_params}" '
        f"--SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --SiftExtraction.gpu_index 0 "
        "--ImageReader.single_camera 1 "
        f"--database_path {db} --image_path {imgdir}",
        skip=skip_sparse,
    )
    # step2: match sift
    do_system(
        f"colmap {matcher}_matcher --SiftMatching.guided_matching=true --database_path {db} --SiftMatching.gpu_index 0 ",
        skip=skip_sparse,
    )

    sparse = os.path.join(output, "sparse")
    os.makedirs(sparse, exist_ok=True)
    # step3: sfm
    do_system(
        f"colmap mapper --database_path {db} --image_path {imgdir} --output_path {sparse}",
        skip=skip_sparse,
    )

    # step3: bundle adjustment
    do_system(
        f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1",
        skip=skip_sparse,
    )
    do_system(
        f"colmap model_converter --input_path {sparse}/0 --output_path {sparse}/0 --output_type TXT",
        skip=skip_sparse,
    )
    do_system(
        f"colmap model_converter --input_path {sparse}/0 --output_path {sparse}/0/points3D.ply --output_type PLY",
        skip=skip_sparse,
    )

    undistort_imgdir = os.path.join(sparse, "0/images")
    if not os.path.exists(undistort_imgdir) and do_undistort_image:
        do_system(
            f"colmap image_undistorter --input_path {sparse}/0 --image_path {imgdir} --output_path {sparse}/0"
        )

    if not do_dense:
        return output

    # step4: patch match
    do_system(
        f"colmap patch_match_stereo --workspace_path {sparse}",
    )

    # step5: depth fusion
    ply_path = os.path.join(output, "pointcloud.ply")
    do_system(
        f"colmap stereo_fusion --workspace_path {sparse} --workspace_format COLMAP --input_type geometric --output_path {ply_path}"
    )

    return output


@otn_manager.NODE.register(name="colmap_to_openmvs")
def colmap_to_openmvs(
    path: str,
    imgdir: str = None,
    output: str = "mvs_output",
    cache: bool = False,
    remove_cache: bool = False,
    mvs_bin_root: str = "/usr/local/bin/OpenMVS",
):
    """Colmap to openmvs.

    https://github.com/cdcseacave/openMVS/wiki/Usage

    Args:
        path: colmap output
        imgdir: image folder
        output: openmvs workspace and output
        cache: cache flag
        mvs_bin_root: OpenMVS bin path
    """
    if os.path.exists(output):
        if cache:
            typer.echo(
                typer.style(f"Found {output}, skip", fg=typer.colors.WHITE)
            )
            return output
        if remove_cache:
            shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    if not os.path.exists(mvs_bin_root):
        typer.echo(
            typer.style(
                f"{mvs_bin_root} not found, forget to install OpenMVS?",
                fg=typer.colors.RED,
            )
        )
        sys.exit()

    if imgdir is None:
        imgdir = os.path.join(path, "sparce/0/images")

    path = os.path.abspath(path)
    imgdir = os.path.abspath(imgdir)

    # create softlink in output
    imgdir_link = os.path.join(output, "images")
    if not os.path.exists(imgdir_link):
        do_system(f"ln -s {imgdir} {imgdir_link}")
    sparse_link = os.path.join(output, "sparse")
    if not os.path.exists(sparse_link):
        do_system(f"ln -s {path}/sparse/0 {sparse_link}")

    # create PINHOLE cameras.txt if not found.
    cameras_txt = f"{sparse_link}/cameras.txt"
    intrinsic = CameraIntrinsic.from_colmap_txt(cameras_txt)
    if intrinsic.camera_model != "PINHOLE":
        do_system(f"mv {cameras_txt} {cameras_txt}.bakup")
    with open(cameras_txt, "w") as fid:
        fid.write(
            """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
"""
        )
        fx, fy, cx, cy = intrinsic.tolist()
        fid.write(f"1 PINHOLE {intrinsic.w} {intrinsic.h} {fx} {fy} {cx} {cy}")

    # step1: interface colmap
    do_system(
        f"{mvs_bin_root}/InterfaceCOLMAP --working-folder {output} --input-file . --output-file model_colmap.mvs",
        skip=os.path.exists(f"{output}/model_colmap.mvs"),
    )

    # step2: densify pointcloud
    do_system(
        f"{mvs_bin_root}/DensifyPointCloud --working-folder {output} --input-file model_colmap.mvs --output-file model_dense.mvs --archive-type -1",
        skip=os.path.exists(f"{output}/model_dense.mvs"),
    )
    do_system(
        f"mkdir -p {output}/dmap_files && mv {output}/depth*.dmap {output}/dmap_files",
        skip=os.path.exists(f"{output}/dmap_files"),
    )

    # step3: reconstruct mesh
    do_system(
        f"{mvs_bin_root}/ReconstructMesh --working-folder {output} --input-file model_dense.mvs --output-file model_dense_mesh.ply",
        skip=os.path.exists(f"{output}/model_dense_mesh.ply"),
    )

    # step4: refine mesh
    do_system(
        f"{mvs_bin_root}/RefineMesh --working-folder {output} --resolution-level 1 "
        f"--input-file model_dense.mvs --mesh-file model_dense_mesh.ply --output-file model_dense_mesh_refine.ply",
        skip=os.path.exists(f"{output}/model_dense_mesh_refine.ply"),
    )

    # step5: texture mesh
    do_system(
        f"{mvs_bin_root}/TextureMesh --working-folder {output} "
        f"--input-file model_dense.mvs --mesh-file model_dense_mesh_refine.ply --output-file model_dense_mesh_refine_texture.ply",
        skip=os.path.exists(f"{output}/model_dense_mesh_refine_texture.ply"),
    )

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output
