"""
Scale recovery infer node for OnePoseviaGen pipeline.
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
from PIL import Image


def _append_repo_paths(repo: Path) -> None:
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


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _patch_untextured_pytorch3d_render() -> None:
    """Give fixture meshes a default material for PyTorch3D rendering."""
    try:
        import torch
        from pytorch3d.io import load_objs_as_meshes
        from pytorch3d.renderer import TexturesVertex
        import oneposeviagen.locate.fit_object_scale as fit_object_scale
    except ImportError:
        return

    def load_with_default_texture(paths, device=None, **kwargs):
        previous_tensor_type = torch.empty(()).type()
        torch.set_default_tensor_type(torch.FloatTensor)
        try:
            mesh = load_objs_as_meshes(paths, device=device, **kwargs)
        finally:
            torch.set_default_tensor_type(previous_tensor_type)
        if mesh.textures is None:
            verts = mesh.verts_padded()
            features = torch.full_like(verts, 0.8)
            mesh.textures = TexturesVertex(verts_features=features)
        return mesh

    fit_object_scale.load_objs_as_meshes = load_with_default_texture


def main(
    model_path: str,
    depth_path: str,
    rgb_path: str,
    masks_path: str,
    repo: str,
    output: str = "scaled_model",
    intrinsics_path: Optional[str] = None,
    refine_iterations: int = 3,
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
        refine_iterations: Number of FoundationPose-guided scale refinement
            iterations after the initial scale estimate.
    """
    repo = Path(repo).expanduser().resolve()
    _append_repo_paths(repo)

    try:
        import trimesh
        import fpose.recover_scale as recover_scale_module
    except ImportError as e:
        raise RuntimeError(
            "OnePoseViaGen scale dependency missing: "
            f"{getattr(e, 'name', str(e))}. Install the official "
            "OnePoseViaGen/FoundationPose environment that provides "
            "`fpose.recover_scale`."
        ) from e
    _patch_untextured_pytorch3d_render()

    # Expand paths
    model_path = Path(model_path).expanduser().resolve()
    depth_path = Path(depth_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()

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

    # Recover scale. OnePoseViaGen utilities resolve checkpoints relative to
    # the repository root.
    with _working_directory(repo):
        scaled_mesh, anchor_pose, final_scale = _recover_scale(
            recover_scale_module,
            str(model_path),
            anchor_depth,
            anchor_rgb,
            anchor_mask,
            str(intrinsic_path),
            "test",
            str(temp_dir),
            refine_iterations=refine_iterations,
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


def _recover_scale(
    recover_scale_module,
    mesh_file: str,
    depth_file: str,
    raw_img: str,
    mask_file: str,
    intrinsic_file: str,
    topic: str,
    out_dir: str,
    *,
    refine_iterations: int,
):
    """Run OnePoseViaGen scale recovery with configurable refinement count."""
    import numpy as np
    import os
    import trimesh

    mesh = trimesh.load(mesh_file, force="mesh")
    mesh_rotated = mesh.copy()

    scales = []
    poses = []

    get_scale = recover_scale_module.get_scale
    get_single_pose = recover_scale_module.get_single_pose

    _, intrinsic_numpy, scale = get_scale(
        mesh_file, depth_file, raw_img, mask_file, out_dir, intrinsic_file
    )
    scales.append(scale)
    scale_matrix = np.array(
        [
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1],
        ]
    )
    mesh.apply_transform(scale_matrix)
    mesh_rotated.apply_transform(scale_matrix)

    anchor_pose = np.eye(4)
    for index in range(refine_iterations):
        anchor_pose = get_single_pose(
            raw_img,
            depth_file,
            mask_file,
            intrinsic_numpy,
            mesh_rotated,
            topic,
            index,
            debug=0,
            est_refine_iter=3,
        )
        poses.append(anchor_pose)

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = anchor_pose[0:3, 0:3]
        mesh_rotated.apply_transform(rotation_matrix)
        rotated_mesh_path = os.path.join(out_dir, "rotated_model.obj")
        mesh_rotated.export(rotated_mesh_path)

        scale_output = os.path.join(out_dir, f"iteration_{index + 1}")
        os.makedirs(scale_output, exist_ok=True)
        _, intrinsic_numpy, scale = get_scale(
            rotated_mesh_path,
            depth_file,
            raw_img,
            mask_file,
            scale_output,
            intrinsic_file,
        )
        if 0.5 <= scale <= 1.5:
            scales.append(scale)
            scale_matrix = np.array(
                [
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1],
                ]
            )
            mesh.apply_transform(scale_matrix)
            mesh_rotated.apply_transform(scale_matrix)

    final_scale = 1.0
    for scale in scales:
        final_scale *= scale
    return mesh, anchor_pose, final_scale


if __name__ == "__main__":
    typer.run(main)
