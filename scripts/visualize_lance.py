#!/usr/bin/env python3
"""Visualize annotations from a Lance dataset. (AI)

This script reads an annotated Lance dataset and generates visualization
collages showing the original images along with their annotations (caption,
tags, detection, segmentation, depth).

The script processes the dataset in batches to avoid OOM issues and supports
configurable stride for selective visualization.

Usage:
    # Basic usage: visualize all images
    python scripts/visualize_lance.py <input_dataset> --output <output_dir>

    # Visualize every 10th image with batch size of 100
    python scripts/visualize_lance.py <input_dataset> --output <output_dir> \\
        --stride 10 --batch-size 100
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence

from pdebug.otn.infer import internimage_semseg as semseg_module
from pdebug.otn.infer.lance_utils import (
    LanceBatch,
    decode_depth_map,
    decode_depth_map_uint16,
    load_lance_batch,
)
import cv2
import numpy as np
import typer
from loguru import logger

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore[assignment]


# Import visualization functions from use_with_lance.py
# We import these functions to reuse the visualization logic
import sys
from pathlib import Path

# Add parent directory to path to import from use_with_lance
_script_dir = Path(__file__).parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Import visualization functions from use_with_lance
# Note: These are private functions but we reuse them for visualization
from use_with_lance import (  # type: ignore[import-untyped]
    _append_text_panel,
    _build_reader_kwargs,
    _combine_visualizations,
    _iter_lance_batches,
    _preview_rows,
    _render_caption,
    _render_depth,
    _render_detection,
    _render_error,
    _render_original,
    _render_segmentation,
    _render_tags,
)


app = typer.Typer(
    help="Generate visualizations from an annotated Lance dataset."
)


def _visualize_models(
    dataset_path: Path,
    *,
    image_col: str,
    reader_kwargs: Dict[str, object],
    output_dir: Path,
    stride: int,
    batch_size: int,
    timestamp_col: Optional[str],
    video_id_col: Optional[str],
    frame_num_col: Optional[str],   
    trigger: Optional[str],
) -> None:
    """Visualize models from Lance dataset.

    This function processes the dataset in batches to avoid OOM issues.
    It only visualizes images at the specified stride interval.

    Args:
        dataset_path: Path to the Lance dataset
        image_col: Name of the image column
        reader_kwargs: Reader configuration
        output_dir: Output directory for visualizations
        stride: Interval between visualized images (1 = all, 2 = every other, etc.)
        batch_size: Number of images to load per batch
        timestamp_col: Optional timestamp column name
        video_id_col: Optional video ID column name
        frame_num_col: Optional frame number column name
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track global index across all batches
    global_index = 0
    visualized_count = 0

    # Create iterator for batches
    batch_iterator = _iter_lance_batches(
        dataset_path,
        image_col=image_col,
        reader_kwargs=reader_kwargs,
        batch_size=batch_size,
        timestamp_col=timestamp_col,
        video_id_col=video_id_col,
        frame_num_col=frame_num_col,
        trigger=trigger,
    )

    for batch_index, batch in enumerate(batch_iterator, start=1):
        if not batch.images:
            continue

        logger.info(
            f"Processing batch {batch_index}: {len(batch.images)} images "
            f"(total visualized: {visualized_count})"
        )

        for local_index, entry in enumerate(batch.metadata):
            # Apply stride: only visualize if global_index % stride == 0
            if global_index % stride != 0:
                global_index += 1
                continue

            image = entry.get("image")
            if not isinstance(image, np.ndarray):
                logger.warning(
                    f"Skipping row {global_index}: image is missing or invalid"
                )
                global_index += 1
                continue

            try:
                visualizations = [
                    _render_original(image),
                    _render_caption(image, entry.get("cortexia_caption")),
                    _render_tags(image, entry.get("cortexia_tags")),
                    _render_detection(image, entry.get("cortexia_detection")),
                    _render_segmentation(
                        image, entry.get("cortexia_segmentation")
                    ),
                    _render_depth(image, entry.get("cortexia_depth")),
                ]
            except Exception as exc:  # pragma: no cover - defensive fallback
                typer.secho(
                    f"Visualization failed on row {global_index}: {exc}",
                    fg=typer.colors.RED,
                )
                visualizations = [
                    _append_text_panel(image, "[Original]", [f"Error: {exc}"]),
                    _render_error(image, "[Caption]", exc),
                    _render_error(image, "[Listing]", exc),
                    _render_error(image, "[Detection]", exc),
                    _render_error(image, "[Segmentation]", exc),
                    _render_error(image, "[Depth]", exc),
                ]

            combined = _combine_visualizations(visualizations)
            frame_name = entry.get("image_name") or f"frame_{global_index:06d}"
            safe_name = Path(str(frame_name)).stem
            output_path = output_dir / f"{global_index:04d}_{safe_name}.png"
            bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(
                str(output_path), bgr
            ):  # pragma: no cover - cv2 returns bool
                raise RuntimeError(
                    f"Failed to write visualization to {output_path}"
                )

            visualized_count += 1
            global_index += 1

        # Clean up batch to free memory
        del batch
        import gc

        gc.collect()

    if visualized_count == 0:
        raise RuntimeError(
            "No images were visualized. Check stride and dataset content."
        )

    logger.info(f"Total visualizations created: {visualized_count}")


@app.command()
def main(
    input_dataset: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
        help="Path to the annotated Lance dataset.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
        help="Directory where visualization collages will be saved.",
    ),
    image_col: str = typer.Option(
        "camera_left",
        "--image-col",
        help="Name of the Lance column that stores image bytes.",
    ),
    timestamp_col: Optional[str] = typer.Option(
        None,
        "--timestamp-col",
        help="Optional timestamp column name.",
    ),
    video_id_col: Optional[str] = typer.Option(
        None,
        "--video-id-col",
        help="Optional video identifier column name.",
    ),
    frame_num_col: Optional[str] = typer.Option(
        None,
        "--frame-num-col",
        help="Optional frame number column name.",
    ),
    row_limit: Optional[int] = typer.Option(
        None,
        "--row-limit",
        min=1,
        help="Limit the number of rows processed from the dataset.",
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        min=1,
        help="Interval between visualized images. 1=all, 2=every other, etc.",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        min=1,
        help="Number of images to load per batch to avoid OOM.",
    ),
    preview: bool = typer.Option(
        True,
        "--preview/--no-preview",
        help="Preview first 3 rows of the dataset before visualization.",
    ),
    trigger: Optional[str] = typer.Option(
        None,
        "--trigger",
        help="Trigger to filter the dataset.",
    ),
) -> None:
    """Generate visualizations from an annotated Lance dataset.

    This script reads an annotated Lance dataset (typically created by
    use_with_lance.py) and generates visualization collages showing:
    - Original image
    - Caption (from Moondream)
    - Tags/Listing (from Qwen2.5-VL)
    - Detection (from GroundingDINO)
    - Segmentation (from InternImage)
    - Depth (from ML-Depth-Pro)

    Each visualization is saved as a PNG file in the output directory.
    """
    dataset_path = input_dataset.resolve()
    output_dir = output_dir.resolve()

    if not dataset_path.exists():
        raise typer.BadParameter(
            f"Input dataset not found at {dataset_path}"
        )

    reader_kwargs = _build_reader_kwargs(
        timestamp_col=timestamp_col,
        video_id_col=video_id_col,
        frame_num_col=frame_num_col,
        row_limit=row_limit,
    )

    # Preview mode: load a small sample for preview
    if preview:
        logger.info(f"Loading sample from Lance dataset for preview...")
        preview_kwargs = reader_kwargs.copy()
        preview_kwargs["row_limit"] = 3
        preview_batch = load_lance_batch(
            dataset_path,
            image_col=image_col,
            reader_kwargs=preview_kwargs,
        )
        if preview_batch.images:
            preview_table = preview_batch.table.combine_chunks()
            _preview_rows(preview_table, limit=3)
            del preview_batch, preview_table
        else:
            logger.warning("No images found for preview")

    logger.info(
        f"Generating visualizations in {output_dir} "
        f"(stride={stride}, batch_size={batch_size})..."
    )
    _visualize_models(
        dataset_path,
        image_col=image_col,
        reader_kwargs=reader_kwargs,
        output_dir=output_dir,
        stride=stride,
        batch_size=batch_size,
        timestamp_col=timestamp_col,
        video_id_col=video_id_col,
        frame_num_col=frame_num_col,
        trigger=trigger,
    )
    logger.info(f"Visualization collages saved to: {output_dir}")


if __name__ == "__main__":
    app()

