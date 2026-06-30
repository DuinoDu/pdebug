"""
Pose estimation infer node for OnePoseviaGen pipeline.
"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from pdebug.piata import Input
from pdebug.utils.fileio import do_system

import cv2
import numpy as np
import typer


def _append_repo_paths(repo: Path) -> None:
    # FoundationPose/OnePoseViaGen use absolute imports from repo roots. Append
    # paths so installed packages keep precedence.
    for path in (
        repo,
        repo / "oneposeviagen",
        repo / "oneposeviagen" / "fpose",
        repo / "oneposeviagen" / "fpose" / "fpose",
        repo / "FoundationPose",
        repo.parent,
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


def _raise_missing_dependency(missing: str, repo: Path):
    raise RuntimeError(
        "OnePoseViaGen pose dependency missing: "
        f"{missing}. Install the official OnePoseViaGen/FoundationPose "
        "environment, including nvdiffrast and CUDA extensions, for "
        f"repo {repo}."
    )


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main(
    rgb_path: str,
    depth_path: str,
    masks_path: str,
    model_path: str,
    repo: str,
    output: str = "pose_estimation",
    intrinsics_path: Optional[str] = None,
    debug: int = 0,
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    init_min_views: int = 20,
    init_inplane_step: int = 90,
):
    """
    Pose estimation for OnePoseviaGen pipeline using FoundationPose.

    Args:
        rgb_path: Path to RGB image folder
        depth_path: Path to depth image folder
        masks_path: Path to mask folder
        model_path: Path to 3D model (.obj file)
        repo: Path to OnePoseviaGen repository
        output: Output directory for pose results
        intrinsics_path: Path to intrinsics JSON file
        debug: Debug level (0-1)
        est_refine_iter: Number of refinement iterations
    """
    # Expand paths
    rgb_path = Path(rgb_path).expanduser().resolve()
    depth_path = Path(depth_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    output.mkdir(parents=True, exist_ok=True)

    _append_repo_paths(repo)

    try:
        import nvdiffrast.torch as dr
        import trimesh
    except ImportError as e:
        _raise_missing_dependency(getattr(e, "name", str(e)), repo)

    try:
        from foundationpose.estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
        )
        from foundationpose.Utils import draw_posed_3d_box, draw_xyz_axis
    except ImportError as first_error:
        try:
            from estimater import (
                FoundationPose,
                PoseRefinePredictor,
                ScorePredictor,
            )
            from Utils import draw_posed_3d_box, draw_xyz_axis
        except ImportError as second_error:
            missing = getattr(second_error, "name", str(second_error))
            if missing == "estimater":
                missing = getattr(first_error, "name", missing)
            _raise_missing_dependency(missing, repo)

    # Get files
    rgb_files = Input(str(rgb_path), name="imgdir").get_reader().imgfiles
    rgb_files.sort(key=lambda x: int(Path(x).stem))
    depth_files = Input(str(depth_path), name="imgdir").get_reader().imgfiles
    depth_files.sort(key=lambda x: int(Path(x).stem))
    mask_files = Input(str(masks_path), name="imgdir").get_reader().imgfiles
    mask_files.sort(key=lambda x: int(Path(x).stem))

    if len(rgb_files) == 0:
        raise RuntimeError("No RGB files found")
    if len(depth_files) == 0:
        raise RuntimeError("No depth files found")
    if len(mask_files) == 0:
        raise RuntimeError("No mask files found")

    # Load intrinsics
    if intrinsics_path is None:
        # Try to find intrinsics.json in parent directories
        intrinsics_path = rgb_path.parent / "intrinsics.json"
        if not intrinsics_path.exists():
            intrinsics_path = rgb_path / ".." / "intrinsics.json"

    with open(intrinsics_path, "r") as f:
        intrinsics_dict = json.load(f)

    if "cam_K" in intrinsics_dict:
        shared_K = np.asarray(intrinsics_dict["cam_K"], dtype=np.float32)
        intrinsics = [shared_K.copy() for _ in range(len(rgb_files))]
    else:
        intrinsics = []
        for i in range(len(rgb_files)):
            K = intrinsics_dict[str(i)]
            intrinsics.append(np.asarray(K, dtype=np.float32))

    # Load 3D model
    mesh = trimesh.load(str(model_path), force="mesh")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces),
        vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        visual=mesh.visual,
        process=False,
    )

    # Setup FoundationPose
    debug_dir = output / "debug"
    debug_dir.mkdir(exist_ok=True)

    with _working_directory(repo):
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(debug_dir),
            debug=debug,
            glctx=glctx,
        )
        est.diameter = float(est.diameter)
        est.make_rotation_grid(
            min_n_views=init_min_views, inplane_step=init_inplane_step
        )

        # Estimate poses
        poses = []
        pose_scores = []
        initialized = False

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        pose_dir = output / "pose_viz"
        pose_dir.mkdir(exist_ok=True)
        video_writer = None

        for frame_id, rgb_file in enumerate(rgb_files):
            color = cv2.imread(str(rgb_file))
            depth = (
                cv2.imread(str(depth_files[frame_id]), -1) / 1e3
            )  # Convert mm to meters
            mask = cv2.imread(str(mask_files[frame_id]), -1)
            K = np.asarray(intrinsics[frame_id], dtype=np.float32)

            # Handle mask format
            if len(mask.shape) == 3:
                for c in range(3):
                    if mask[..., c].sum() > 0:
                        mask = mask[..., c]
                        break
            mask = mask.astype(bool)

            if not initialized:
                pose = est.register(
                    K=K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=est_refine_iter,
                )
                initialized = True
            else:
                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                )
            poses.append(pose.reshape(4, 4))
            pose_scores.append(float(frame_id == 0))

            # Create visualization
            center_pose = pose @ np.linalg.inv(to_origin)
            rgb_vis = draw_posed_3d_box(
                K, img=color, ob_in_cam=center_pose, bbox=bbox
            )
            rgb_vis = draw_xyz_axis(
                rgb_vis,
                ob_in_cam=center_pose,
                scale=0.1,
                K=K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            pose_img_path = pose_dir / f"pose_{frame_id:06d}.jpg"
            cv2.imwrite(str(pose_img_path), rgb_vis)
            if video_writer is None:
                h, w = rgb_vis.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output / "pose_video.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (w, h),
                )
            video_writer.write(rgb_vis)

    # Save poses
    poses_dict = {}
    for frame_id in range(len(poses)):
        poses_dict[str(frame_id)] = {
            "T_cam_obj": poses[frame_id].tolist(),
            "score": pose_scores[frame_id],
        }

    poses_file = output / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_dict, f, indent=2)

    if video_writer is not None:
        video_writer.release()

    typer.echo(
        typer.style(
            f"Pose estimation completed. Results saved to {output}",
            fg=typer.colors.GREEN,
        )
    )
    typer.echo(
        typer.style(f"Estimated {len(poses)} poses", fg=typer.colors.GREEN)
    )

    return str(output)


if __name__ == "__main__":
    typer.run(main)
