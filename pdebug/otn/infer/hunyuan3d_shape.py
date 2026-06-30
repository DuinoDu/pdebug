"""
Hunyuan3D-Shape infer node for 3D mesh generation from images.

Depends:
    torch, torchvision, diffusers, transformers
"""
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from pdebug.piata import Input, Output
from pdebug.visp import draw

import numpy as np
import typer
from PIL import Image


def _extract_mesh(pipeline_output):
    """Normalize Hunyuan3D pipeline outputs across upstream versions."""
    if hasattr(pipeline_output, "mesh"):
        pipeline_output = pipeline_output.mesh
    if isinstance(pipeline_output, (list, tuple)):
        if not pipeline_output:
            return None
        return pipeline_output[0]
    return pipeline_output


def hunyuan3d_shape_main(
    repo: str,
    input_path: str,
    output: str = "tmp_hunyuan3d_shape",
    model_path: str = "tencent/Hunyuan3D-2.1",
    octree_resolution: int = 256,
    num_inference_steps: int = 5,
    guidance_scale: float = 5.0,
    num_chunks: int = 8000,
    seed: int = 1234,
    cache: bool = True,
):
    """
    Generate 3D mesh from single image using Hunyuan3D-Shape model.

    Args:
        repo: Path to Hunyuan3D repository
        input_path: Path to input image file or folder containing images
        output: Output directory for generated 3D meshes
        model_path: Path to pretrained Hunyuan3D-Shape model
        octree_resolution: Resolution of the octree for mesh generation (64-512)
        num_inference_steps: Number of diffusion steps (1-20)
        guidance_scale: Guidance scale for generation (0.1-20.0)
        num_chunks: Number of chunks for processing (1000-20000)
        seed: Random seed for reproducible generation
        cache: Whether to skip if output already exists
    """

    # Expand paths
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache and (output / "meshes.json").exists():
        print(f"Output {output} exists, skipping...")
        return str(output)

    try:
        # Add repo to Python path
        sys.path.insert(0, str(repo / "hy3dshape"))
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as e:
        raise ImportError(f"Failed to import Hunyuan3D-Shape pipeline: {e}")

    # Initialize pipeline
    print("Loading Hunyuan3D-Shape pipeline...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, subfolder="hunyuan3d-dit-v2-1"
    )

    # Get input images
    if input_path.is_dir():
        image_files = (
            list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
        )
        image_files.sort(key=lambda x: x.stem)
    else:
        image_files = [input_path]

    if not image_files:
        raise ValueError("No valid image files found")

    print(f"Found {len(image_files)} images to process")

    # Process each image
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path}")

        # Load and preprocess image
        image = Image.open(img_path).convert("RGBA")

        # Generate mesh
        mesh = _extract_mesh(
            shape_pipeline(
                image=image,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_chunks=num_chunks,
                seed=seed + i,
            )
        )
        if mesh is None:
            raise RuntimeError(
                "Hunyuan3D-Shape returned no mesh; increase "
                "num_inference_steps or use an object-focused RGBA image."
            )

        # Save mesh
        output_mesh_path = output / f"{img_path.stem}_shape.glb"
        mesh.export(str(output_mesh_path))

        results.append(
            {
                "input_image": str(img_path),
                "output_mesh": str(output_mesh_path),
                "parameters": {
                    "octree_resolution": octree_resolution,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "num_chunks": num_chunks,
                    "seed": seed + i,
                },
            }
        )

        print(f"Saved mesh to {output_mesh_path}")

    # Save metadata
    with open(output / "meshes.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(
        typer.style(
            f"Successfully generated {len(results)} 3D meshes in {output}",
            fg=typer.colors.GREEN,
        )
    )
    return str(output)


if __name__ == "__main__":
    typer.run(hunyuan3d_shape_main)
