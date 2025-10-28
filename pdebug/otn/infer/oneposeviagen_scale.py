"""
Scale recovery infer node for OnePoseviaGen pipeline.
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
from PIL import Image


@otn_manager.NODE.register(name="oneposeviagen_scale")
def main(
    model_path: str,
    depth_path: str,
    rgb_path: str,
    masks_path: str,
    repo: str,
    output: str = "scaled_model",
    intrinsics_path: Optional[str] = None,
):
    """
    Scale recovery for OnePoseviaGen pipeline.

    Args:
        model_path: Path to 3D model (.obj file)
        depth_path: Path to depth image folder
        rgb_path: Path to RGB image folder
        masks_path: Path to mask folder
        repo: Path to OnePoseviaGen repository
        output: Output directory for scaled model
        intrinsics_path: Path to intrinsics JSON file
    """
    try:
        import trimesh
        from fpose.recover_scale import recover_scale
    except ImportError as e:
        print(f"Warning: Required modules not available: {e}")
        recover_scale = None

    if recover_scale is None:
        raise RuntimeError(
            "Required modules not installed. Please install OnePoseviaGen."
        )

    # Expand paths
    model_path = Path(model_path).expanduser().resolve()
    depth_path = Path(depth_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
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

    # Use first frame as anchor
    anchor_rgb = str(rgb_files[0])
    anchor_depth = str(depth_files[0])
    anchor_mask = str(mask_files[0])

    # Load intrinsics
    if intrinsics_path is None:
        # Try to find intrinsics.json in parent directories
        intrinsics_path = rgb_path.parent / "intrinsics.json"
        if not intrinsics_path.exists():
            intrinsics_path = rgb_path / ".." / "intrinsics.json"

    with open(intrinsics_path, "r") as f:
        intrinsics_dict = json.load(f)

    # Use intrinsics for first frame
    anchor_intrinsic = np.array(intrinsics_dict["0"])

    # Create temporary directory for intermediate files
    temp_dir = output / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Save intrinsic to file
    intrinsic_path = temp_dir / "intrinsic.txt"
    np.savetxt(str(intrinsic_path), anchor_intrinsic, fmt="%.6f")

    # Recover scale
    scaled_mesh, anchor_pose, final_scale = recover_scale(
        str(model_path),
        anchor_depth,
        anchor_rgb,
        anchor_mask,
        str(intrinsic_path),
        "test",
        str(temp_dir),
    )

    # Save scaled model
    scaled_model_path = output / "scaled_model.obj"
    scaled_mesh.export(str(scaled_model_path))

    # Save high resolution scaled model
    high_mesh = trimesh.load(str(model_path))
    high_mesh.vertices = high_mesh.vertices * final_scale
    high_mesh_path = output / "high_scaled_model.obj"
    high_mesh.export(str(high_mesh_path))

    # Save scale info
    scale_info = {
        "scale_factor": float(final_scale),
        "anchor_pose": anchor_pose.tolist()
        if hasattr(anchor_pose, "tolist")
        else anchor_pose,
        "original_model": str(model_path),
        "scaled_model": str(scaled_model_path),
        "high_scaled_model": str(high_mesh_path),
        "anchor_rgb": anchor_rgb,
        "anchor_depth": anchor_depth,
        "anchor_mask": anchor_mask,
    }

    with open(output / "scale_info.json", "w") as f:
        json.dump(scale_info, f, indent=2)

    typer.echo(
        typer.style(
            f"Scale recovery completed. Scale factor: {final_scale}",
            fg=typer.colors.GREEN,
        )
    )
    typer.echo(
        typer.style(
            f"Scaled model saved to: {scaled_model_path}",
            fg=typer.colors.GREEN,
        )
    )

    return str(output)


if __name__ == "__main__":
    typer.run(main)
