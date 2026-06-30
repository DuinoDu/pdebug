"""
3D model generation infer node for OnePoseviaGen pipeline.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

from pdebug.piata import Input
from pdebug.utils.fileio import do_system

import cv2
import numpy as np
import typer
from PIL import Image


def _append_repo_path(repo: Path) -> None:
    for path in (
        repo,
        repo / "oneposeviagen",
        repo / "oneposeviagen" / "trellis",
        repo / "oneposeviagen" / "Amodal3R",
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.append(path_str)


def _missing_dependency_error(missing: str, repo: Path) -> RuntimeError:
    return RuntimeError(
        "OnePoseViaGen 3D generation dependency missing: "
        f"{missing}. Install the official OnePoseViaGen environment and "
        f"download checkpoints under {repo}/checkpoints/OnePoseViaGen."
    )


def _patch_timm_layers_compat() -> None:
    """Expose new timm layer imports when an older timm is installed."""
    try:
        import timm.layers as timm_layers
        import timm.models.layers as model_layers
    except ImportError:
        return
    for name in (
        "DropPath",
        "get_act_layer",
        "lecun_normal_",
        "resample_abs_pos_embed",
        "to_2tuple",
        "trunc_normal_",
    ):
        if not hasattr(timm_layers, name) and hasattr(model_layers, name):
            setattr(timm_layers, name, getattr(model_layers, name))


def _move_pipeline_to_cuda(pipeline) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "OnePoseViaGen 3D generation requires PyTorch."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("OnePoseViaGen 3D generation requires CUDA.")
    pipeline.to(torch.device("cuda"))


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
    preview_resolution: int = 1024,
    preview_num_frames: int = 120,
    preview_fps: int = 15,
    generate_gaussian: bool = True,
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
        preview_resolution: Resolution for the preview render
        preview_num_frames: Number of preview frames to render
        preview_fps: Preview video frames per second
        generate_gaussian: Whether to decode Gaussian output and run
            Gaussian-backed postprocessing.
    """
    repo = Path(repo).expanduser().resolve()
    _append_repo_path(repo)
    _patch_timm_layers_compat()

    try:
        from amodal3r.pipelines import Amodal3RImageTo3DPipeline
        from amodal3r.utils import postprocessing_utils, render_utils
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import (
            postprocessing_utils as postprocessing_utils_hi3dgen,
        )
        from trellis.utils import render_utils as render_utils_hi3dgen
    except ImportError as e:
        missing = getattr(e, "name", str(e))
        raise _missing_dependency_error(missing, repo) from e

    # Expand paths
    rgb_path = Path(rgb_path).expanduser().resolve()
    masks_path = Path(masks_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()

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
    formats = ["mesh", "gaussian"] if generate_gaussian else ["mesh"]

    if is_occluded or model_type == "amodal3r":
        # Use Amodal3R
        pipeline = Amodal3RImageTo3DPipeline.from_pretrained(
            repo / "checkpoints" / "OnePoseViaGen" / "Amodal3R"
        )
        _move_pipeline_to_cuda(pipeline)

        outputs = pipeline.run_multi_image(
            [processed_rgb],
            [final_mask],
            seed=seed,
            formats=formats,
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
        mesh_path = model_dir / "model.obj"
        if generate_gaussian:
            generated_gs = outputs["gaussian"][0]
            video_geo = render_utils.render_video(
                generated_gs,
                resolution=preview_resolution,
                num_frames=preview_num_frames,
            )["color"]
            trimesh_mesh = postprocessing_utils.to_glb(
                generated_gs, generated_mesh, verbose=False
            )
        else:
            video_geo = None
            trimesh_mesh = _mesh_result_to_trimesh(generated_mesh)

    else:
        # Use Hi3DGen
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            repo / "checkpoints" / "OnePoseViaGen" / "Hi3DGen_Color"
        )
        _move_pipeline_to_cuda(pipeline)

        outputs = pipeline.run(
            processed_rgb,
            seed=seed,
            formats=formats,
            preprocess_image=False,
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

        # Save video and mesh
        video_geo = render_utils_hi3dgen.render_video(
            generated_mesh,
            resolution=preview_resolution,
            num_frames=preview_num_frames,
        )["color"]
        mesh_path = model_dir / "model.obj"
        if generate_gaussian:
            generated_gs = outputs["gaussian"][0]
            trimesh_mesh = postprocessing_utils_hi3dgen.to_trimesh(
                generated_gs, generated_mesh, verbose=False
            )
        else:
            trimesh_mesh = _mesh_result_to_trimesh(generated_mesh)

    # Save preview video
    import imageio

    preview_path = model_dir / "preview.mp4"
    if video_geo is not None:
        imageio.mimsave(str(preview_path), video_geo, fps=preview_fps)
    else:
        preview_path = None

    # Save mesh
    import trimesh

    # Rotate to correct orientation
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    trimesh_mesh.apply_transform(R)
    _safe_export_obj(trimesh_mesh, mesh_path)

    # Save high resolution mesh
    high_mesh_path = model_dir / "high_mesh.obj"
    _save_mesh(generated_mesh, high_mesh_path)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "is_occluded": is_occluded,
        "seed": seed,
        "mesh_path": str(mesh_path),
        "high_mesh_path": str(high_mesh_path),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "generate_gaussian": generate_gaussian,
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


def _mesh_result_to_trimesh(mesh_result):
    import trimesh

    vertices = (
        mesh_result.vertices.detach().cpu().numpy()
        if hasattr(mesh_result.vertices, "detach")
        else mesh_result.vertices
    )
    faces = (
        mesh_result.faces.detach().cpu().numpy()
        if hasattr(mesh_result.faces, "detach")
        else mesh_result.faces
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vertex_attrs = getattr(mesh_result, "vertex_attrs", None)
    if vertex_attrs is not None:
        attrs = (
            vertex_attrs.detach().cpu().numpy()
            if hasattr(vertex_attrs, "detach")
            else vertex_attrs
        )
        mesh.visual.vertex_colors = attrs
    return mesh


def _safe_export_obj(mesh, filename: Path | str) -> None:
    """Write OBJ text directly to avoid native exporter crashes."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    with open(filename, "w", encoding="utf-8") as file:
        file.write("# Generated by pdebug OnePoseViaGen integration\n")
        for vertex in vertices:
            file.write(
                f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n"
            )
        for face in faces:
            if len(face) == 3:
                file.write(
                    f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n"
                )
            else:
                indices = " ".join(str(int(index) + 1) for index in face)
                file.write(f"f {indices}\n")


def _save_mesh(mesh_result, filename: Path | str):
    """Save mesh to file."""
    mesh = _mesh_result_to_trimesh(mesh_result)
    _safe_export_obj(mesh, filename)


if __name__ == "__main__":
    typer.run(main)
