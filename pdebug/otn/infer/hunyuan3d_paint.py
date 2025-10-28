"""
Hunyuan3D-Paint infer node for PBR texture generation on 3D meshes.

Depends:
    torch, torchvision, diffusers, transformers, trimesh
"""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.visp import draw

import numpy as np
import typer
from PIL import Image


@otn_manager.NODE.register(name="hunyuan3d_paint")
def hunyuan3d_paint_main(
    repo: str,
    mesh_path: str,
    image_path: str,
    output: str = "tmp_hunyuan3d_paint",
    model_path: str = "tencent/Hunyuan3D-2.1",
    max_num_view: int = 6,
    resolution: int = 512,
    face_count: int = 40000,
    cache: bool = True,
):
    """
    Generate PBR textures for 3D meshes using Hunyuan3D-Paint model.

    Args:
        repo: Path to Hunyuan3D repository
        mesh_path: Path to input 3D mesh file (glb/obj/ply) or folder containing meshes
        image_path: Path to reference image(s) for texture generation (file or folder)
        output: Output directory for textured 3D meshes
        model_path: Path to pretrained Hunyuan3D-Paint model
        max_num_view: Maximum number of views for texture generation (6-9)
        resolution: Texture resolution (512 or 768)
        face_count: Maximum number of faces for texture generation (1000-100000)
        cache: Whether to skip if output already exists
    """

    # Expand paths
    mesh_path = Path(mesh_path).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache and (output / "textures.json").exists():
        print(f"Output {output} exists, skipping...")
        return str(output)

    try:
        # Add repo to Python path
        sys.path.insert(0, str(repo))
        from hy3dpaint.textureGenPipeline import (
            Hunyuan3DPaintConfig,
            Hunyuan3DPaintPipeline,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import Hunyuan3D-Paint pipeline: {e}")

    # Initialize pipeline
    print("Loading Hunyuan3D-Paint pipeline...")

    # Configure pipeline
    config = Hunyuan3DPaintConfig(
        max_num_view=max_num_view, resolution=resolution
    )
    config.realesrgan_ckpt_path = str(
        repo / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    )
    config.multiview_cfg_path = str(
        repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
    )
    config.custom_pipeline = "hy3dpaint.hunyuanpaintpbr"
    config.multiview_pretrained_path = model_path

    paint_pipeline = Hunyuan3DPaintPipeline(config)

    # Get mesh files
    if mesh_path.is_dir():
        mesh_files = (
            list(mesh_path.glob("*.glb"))
            + list(mesh_path.glob("*.obj"))
            + list(mesh_path.glob("*.ply"))
        )
        mesh_files.sort(key=lambda x: x.stem)
    else:
        mesh_files = [mesh_path]

    if not mesh_files:
        raise ValueError("No valid mesh files found")

    # Get image files
    if image_path.is_dir():
        image_files = (
            list(image_path.glob("*.png"))
            + list(image_path.glob("*.jpg"))
            + list(image_path.glob("*.jpeg"))
        )
        image_files.sort(key=lambda x: x.stem)
    else:
        image_files = [image_path]

    if not image_files:
        raise ValueError("No valid image files found")

    print(f"Found {len(mesh_files)} meshes and {len(image_files)} images")

    # Ensure we have matching pairs (use modulo if counts don't match)
    results = []
    for i, mesh_file in enumerate(mesh_files):
        img_file = image_files[i % len(image_files)]
        print(
            f"Processing mesh {i+1}/{len(mesh_files)}: {mesh_file.name} with image: {img_file.name}"
        )

        # Load reference image
        reference_image = Image.open(img_file).convert("RGB")

        # Generate texture
        output_mesh_path = output / f"{mesh_file.stem}_textured.glb"
        textured_mesh_path = paint_pipeline(
            mesh_path=str(mesh_file),
            image_path=str(img_file),
            output_mesh_path=str(output_mesh_path),
            face_count=face_count,
        )

        results.append(
            {
                "input_mesh": str(mesh_file),
                "reference_image": str(img_file),
                "output_textured_mesh": str(textured_mesh_path),
                "parameters": {
                    "max_num_view": max_num_view,
                    "resolution": resolution,
                    "face_count": face_count,
                },
            }
        )

        print(f"Saved textured mesh to {textured_mesh_path}")

    # Save metadata
    with open(output / "textures.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(
        typer.style(
            f"Successfully generated textures for {len(results)} meshes in {output}",
            fg=typer.colors.GREEN,
        )
    )
    return str(output)


if __name__ == "__main__":
    typer.run(hunyuan3d_paint_main)
