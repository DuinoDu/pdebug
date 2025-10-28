"""
Hunyuan3D Background Remover infer node for removing backgrounds from images.

Depends:
    torch, torchvision, PIL
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


@otn_manager.NODE.register(name="hunyuan3d_rembg")
def hunyuan3d_rembg_main(
    repo: str,
    input_path: str,
    output: str = "tmp_hunyuan3d_rembg",
    output_format: str = "RGBA",
    cache: bool = True,
):
    """
    Remove background from images using Hunyuan3D background remover.

    Args:
        repo: Path to Hunyuan3D repository
        input_path: Path to input image file or folder containing images
        output: Output directory for background-removed images
        output_format: Output image format (RGBA, RGB, PNG, JPEG)
        cache: Whether to skip if output already exists
    """

    # Expand paths
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache and (output / "background_removed.json").exists():
        print(f"Output {output} exists, skipping...")
        return str(output)

    try:
        # Add repo to Python path
        sys.path.insert(0, str(repo / "hy3dshape"))
        from hy3dshape.rembg import BackgroundRemover
    except ImportError as e:
        raise ImportError(
            f"Failed to import Hunyuan3D background remover: {e}"
        )

    # Initialize background remover
    print("Loading background remover...")
    rembg = BackgroundRemover()

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

    # Determine file extension based on output format
    if output_format.upper() == "JPEG" or output_format.upper() == "JPG":
        file_ext = ".jpg"
    else:
        file_ext = ".png"

    # Process each image
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_path}")

        # Load image
        image = Image.open(img_path)

        # Remove background
        if image.mode == "RGB":
            # Already RGB, convert to RGBA for background removal
            image_rgba = image.convert("RGBA")
            processed_image = rembg(image_rgba)
        else:
            # Already has alpha or other format
            processed_image = rembg(image)

        # Convert to desired output format
        if output_format.upper() == "RGB":
            # Create white background for RGB output
            background = Image.new(
                "RGB", processed_image.size, (255, 255, 255)
            )
            if processed_image.mode == "RGBA":
                background.paste(
                    processed_image, mask=processed_image.split()[-1]
                )
            else:
                background.paste(processed_image)
            final_image = background
        elif output_format.upper() == "JPEG" or output_format.upper() == "JPG":
            # JPEG doesn't support transparency, use white background
            background = Image.new(
                "RGB", processed_image.size, (255, 255, 255)
            )
            if processed_image.mode == "RGBA":
                background.paste(
                    processed_image, mask=processed_image.split()[-1]
                )
            else:
                background.paste(processed_image)
            final_image = background
        else:  # RGBA or PNG
            final_image = processed_image

        # Save processed image
        output_filename = f"{img_path.stem}_no_bg{file_ext}"
        output_path = output / output_filename
        final_image.save(str(output_path))

        results.append(
            {
                "input_image": str(img_path),
                "output_image": str(output_path),
                "original_mode": image.mode,
                "output_format": output_format,
                "output_size": final_image.size,
            }
        )

        print(f"Saved background-removed image to {output_path}")

    # Save metadata
    with open(output / "background_removed.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(
        typer.style(
            f"Successfully removed backgrounds from {len(results)} images in {output}",
            fg=typer.colors.GREEN,
        )
    )
    return str(output)


if __name__ == "__main__":
    typer.run(hunyuan3d_rembg_main)
