"""
3D model generation infer node for OnePoseviaGen pipeline.
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


@otn_manager.NODE.register(name="oneposeviagen_3dgen")
def main(
    rgb_path: str,
    masks_path: str,
    repo: str,
    output: str = "3d_model",
    seed: int = 42,
    randomize_seed: bool = True,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    slat_guidance_strength: float = 15.0,
    slat_sampling_steps: int = 25,
    is_occluded: bool = False,
    model_type: str = "hi3dgen",
):
    """
    3D model generation for OnePoseviaGen pipeline.

    Args:
        rgb_path: Path to RGB image folder
        masks_path: Path to mask folder
        repo: Path to OnePoseviaGen repository
        output: Output directory for 3D model
        seed: Random seed
        randomize_seed: Whether to randomize seed
        ss_guidance_strength: Sparse structure guidance strength
        ss_sampling_steps: Sparse structure sampling steps
        slat_guidance_strength: SLAT guidance strength
        slat_sampling_steps: SLAT sampling steps
        is_occluded: Whether object is occluded
        model_type: Model type (hi3dgen or amodal3r)
    """
    try:
        from amodal3r.pipelines import Amodal3RImageTo3DPipeline
        from amodal3r.utils import postprocessing_utils, render_utils
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import (
            postprocessing_utils as postprocessing_utils_hi3dgen,
        )
        from trellis.utils import render_utils as render_utils_hi3dgen
    except ImportError as e:
        print(f"Warning: Required modules not available: {e}")
        Amodal3RImageTo3DPipeline = None
        TrellisImageTo3DPipeline = None

    if Amodal3RImageTo3DPipeline is None or TrellisImageTo3DPipeline is None:
        raise RuntimeError(
            "Required modules not installed. Please install OnePoseviaGen."
        )

    # Expand paths
    rgb_path = Path(rgb_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    output.mkdir(parents=True, exist_ok=True)

    # Get RGB and mask files
    rgb_files = Input(str(rgb_path), name="imgdir").get_reader().imgfiles
    rgb_files.sort(key=lambda x: int(Path(x).stem))
    mask_files = Input(str(masks_path), name="imgdir").get_reader().imgfiles
    mask_files.sort(key=lambda x: int(Path(x).stem))

    if len(rgb_files) == 0:
        raise RuntimeError("No RGB files found")
    if len(mask_files) == 0:
        raise RuntimeError("No mask files found")

    # Use first frame for 3D generation
    rgb_image = Image.open(rgb_files[0])
    mask_image = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)

    # Process mask
    if mask_image.max() == 255:
        mask_image = mask_image // 255

    # Create final mask based on occlusion
    if is_occluded:
        # Use Amodal3R for occluded objects
        final_mask = _create_occlusion_mask(mask_image, rgb_image.size)
    else:
        # Use Hi3DGen for non-occluded objects
        final_mask = _create_simple_mask(mask_image)

    final_mask_path = output / "final_mask.png"
    final_mask.save(str(final_mask_path))

    # Process RGB image
    processed_rgb = _process_rgb_image(rgb_image, mask_image)

    # Set random seed
    if randomize_seed:
        seed = np.random.randint(0, 2**32 - 1)

    # Generate 3D model
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)

    if is_occluded or model_type == "amodal3r":
        # Use Amodal3R
        pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
            repo / "checkpoints" / "OnePoseViaGen" / "Amodal3R"
        )

        outputs = pipeline.run_multi_image(
            [processed_rgb],
            [final_mask],
            seed=seed,
            formats=["mesh", "gaussian"],
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        generated_mesh = outputs["mesh"][0]
        generated_gs = outputs["gaussian"][0]

        # Save video and mesh
        video_geo = render_utils.render_video(
            generated_gs, resolution=1024, num_frames=120
        )["color"]
        mesh_path = model_dir / "model.obj"
        trimesh_mesh = postprocessing_utils.to_glb(
            generated_gs, generated_mesh, verbose=False
        )

    else:
        # Use Hi3DGen
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            repo / "checkpoints" / "OnePoseViaGen" / "Hi3DGen_Color"
        )

        outputs = pipeline.run(
            processed_rgb,
            seed=seed,
            formats=["mesh", "gaussian"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        generated_mesh = outputs["mesh"][0]
        generated_gs = outputs["gaussian"][0]

        # Save video and mesh
        video_geo = render_utils_hi3dgen.render_video(
            generated_mesh, resolution=1024, num_frames=120
        )["color"]
        mesh_path = model_dir / "model.obj"
        trimesh_mesh = postprocessing_utils_hi3dgen.to_trimesh(
            generated_gs, generated_mesh, verbose=False
        )

    # Save preview video
    import imageio

    preview_path = model_dir / "preview.mp4"
    imageio.mimsave(str(preview_path), video_geo, fps=15)

    # Save mesh
    import trimesh

    # Rotate to correct orientation
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    trimesh_mesh.apply_transform(R)
    trimesh_mesh.export(str(mesh_path), file_type="obj")

    # Save high resolution mesh
    high_mesh_path = model_dir / "high_mesh.obj"
    _save_mesh(generated_mesh, str(high_mesh_path))

    # Save metadata
    metadata = {
        "model_type": model_type,
        "is_occluded": is_occluded,
        "seed": seed,
        "mesh_path": str(mesh_path),
        "high_mesh_path": str(high_mesh_path),
        "preview_path": str(preview_path),
    }

    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(
        typer.style(f"3D model generated: {mesh_path}", fg=typer.colors.GREEN)
    )
    return str(output)


def _create_simple_mask(mask_image: np.ndarray) -> Image.Image:
    """Create simple mask for non-occluded objects."""
    final_mask = (
        np.ones_like(mask_image, dtype=np.uint8) * 255
    )  # Background white
    final_mask[mask_image > 0] = 188  # Object gray
    return Image.fromarray(final_mask)


def _create_occlusion_mask(
    mask_image: np.ndarray, image_size: tuple
) -> Image.Image:
    """Create occlusion-aware mask for occluded objects."""
    # For now, use simple mask - in real implementation, this would use
    # the occlusion detection logic from OnePoseviaGen
    return _create_simple_mask(mask_image)


def _process_rgb_image(
    rgb_image: Image.Image, mask_image: np.ndarray
) -> Image.Image:
    """Process RGB image with mask."""
    # Convert to numpy
    rgb_np = np.array(rgb_image)

    # Apply mask
    mask_binary = (mask_image > 0).astype(np.uint8) * 255

    # Apply mask to RGB
    result_img = cv2.bitwise_and(rgb_np, rgb_np, mask=mask_binary)

    # Convert back to PIL
    result_pil = Image.fromarray(result_img)

    # Crop and resize
    bbox = np.argwhere(mask_binary > 0)
    if len(bbox) > 0:
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)

        # Add padding
        x_min = max(0, x_min - 20)
        y_min = max(0, y_min - 20)
        x_max = min(rgb_np.shape[1], x_max + 20)
        y_max = min(rgb_np.shape[0], y_max + 20)

        result_pil = result_pil.crop((x_min, y_min, x_max, y_max))

    # Resize to standard size
    result_pil = result_pil.resize((518, 518), Image.LANCZOS)

    return result_pil


def _save_mesh(mesh_result, filename: str):
    """Save mesh to file."""
    import trimesh

    vertices = (
        mesh_result.vertices.cpu().numpy()
        if hasattr(mesh_result.vertices, "cpu")
        else mesh_result.vertices
    )
    faces = (
        mesh_result.faces.cpu().numpy()
        if hasattr(mesh_result.faces, "cpu")
        else mesh_result.faces
    )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if (
        hasattr(mesh_result, "vertex_attrs")
        and mesh_result.vertex_attrs is not None
    ):
        attrs = (
            mesh_result.vertex_attrs.cpu().numpy()
            if hasattr(mesh_result.vertex_attrs, "cpu")
            else mesh_result.vertex_attrs
        )
        mesh.visual.vertex_colors = attrs

    mesh.export(filename)


if __name__ == "__main__":
    typer.run(main)
