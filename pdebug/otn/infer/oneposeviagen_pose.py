"""
Pose estimation infer node for OnePoseviaGen pipeline.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.fileio import do_system

import cv2
import numpy as np
import typer


@otn_manager.NODE.register(name="oneposeviagen_pose")
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
    print("WIP")

    try:
        import dr
        import trimesh
        from fpose.estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
        )
    except ImportError as e:
        print(f"Warning: Required modules not available: {e}")
        ScorePredictor = None
        PoseRefinePredictor = None
        FoundationPose = None

    if ScorePredictor is None or FoundationPose is None:
        raise RuntimeError(
            "Required modules not installed. Please install OnePoseviaGen."
        )

    # Expand paths
    rgb_path = Path(rgb_path).expanduser().resolve()
    depth_path = Path(depth_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    output.mkdir(parents=True, exist_ok=True)

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

    # Get intrinsics for all frames
    intrinsics = [intrinsics_dict[str(i)] for i in range(len(rgb_files))]

    # Load 3D model
    mesh = trimesh.load(str(model_path), force="mesh")

    # Setup FoundationPose
    debug_dir = output / "debug"
    debug_dir.mkdir(exist_ok=True)

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

    # Estimate poses
    poses = []
    rgbs = []
    depths = []

    for frame_id, rgb_file in enumerate(rgb_files):
        color = cv2.imread(str(rgb_file))
        depth = (
            cv2.imread(str(depth_files[frame_id]), -1) / 1e3
        )  # Convert mm to meters
        mask = cv2.imread(str(mask_files[frame_id]), -1)
        K = np.array(intrinsics[frame_id])

        # Handle mask format
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = mask.astype(bool)

        # Estimate pose
        pose = est.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=est_refine_iter,
        )
        poses.append(pose.reshape(4, 4))

        # Create visualization
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        # Draw 3D box and axes
        from fpose.estimater import (
            draw_posed_3d_box_with_depth,
            draw_xyz_axis_with_depth,
        )

        rgb_vis, dep_vis = draw_posed_3d_box_with_depth(
            K, img=color, depth=depth, ob_in_cam=center_pose, bbox=bbox
        )
        rgb_new, dep_new = draw_xyz_axis_with_depth(
            rgb_vis,
            depth=dep_vis,
            ob_in_cam=center_pose,
            scale=0.3,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        rgb_new = np.transpose(rgb_new, (2, 0, 1)) / 255
        rgbs.append(rgb_new)
        depths.append(dep_new)

    # Save poses
    poses_dict = {
        str(frame_id): poses[frame_id].tolist()
        for frame_id in range(len(poses))
    }

    poses_file = output / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_dict, f, indent=2)

    # Save visualization data
    viz_data = {
        "poses": poses,
        "rgbs": rgbs,
        "depths": depths,
        "rgb_files": [str(f) for f in rgb_files],
        "depth_files": [str(f) for f in depth_files],
        "mask_files": [str(f) for f in mask_files],
    }

    # Create pose visualization video
    pose_dir = output / "pose_viz"
    pose_dir.mkdir(exist_ok=True)

    # Save individual pose visualizations
    for i, (rgb_file, pose) in enumerate(zip(rgb_files, poses)):
        color = cv2.imread(str(rgb_file))
        K = np.array(intrinsics[i])

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        center_pose = pose @ np.linalg.inv(to_origin)

        # Draw visualization
        from fpose.estimater import (
            draw_posed_3d_box_with_depth,
            draw_xyz_axis_with_depth,
        )

        rgb_vis, _ = draw_posed_3d_box_with_depth(
            K, img=color, depth=None, ob_in_cam=center_pose, bbox=bbox
        )
        rgb_vis, _ = draw_xyz_axis_with_depth(
            rgb_vis,
            depth=None,
            ob_in_cam=center_pose,
            scale=0.3,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        pose_img_path = pose_dir / f"pose_{i:06d}.jpg"
        cv2.imwrite(str(pose_img_path), rgb_vis)

    # Create pose video
    import imageio

    pose_imgs = []
    for i in range(len(rgb_files)):
        img_path = pose_dir / f"pose_{i:06d}.jpg"
        if img_path.exists():
            pose_imgs.append(cv2.imread(str(img_path)))

    if pose_imgs:
        pose_video = output / "pose_video.mp4"
        imageio.mimsave(str(pose_video), pose_imgs, fps=10)

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
