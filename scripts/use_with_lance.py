#!/usr/bin/env python3
"""Run Cortexia-style pipelines over a Lance dataset using pdebug OTN nodes."""

from __future__ import annotations
import gc
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pdebug.otn.infer import groundingdino_node as groundingdino_module
from pdebug.otn.infer import internimage_semseg as semseg_module
from pdebug.otn.infer import ml_depth_pro_node as ml_depth_pro_module
from pdebug.otn.infer import moondream_node as moondream_module
from pdebug.otn.infer import qwen2_5_vl
from pdebug.otn.infer.lance_utils import (
    bbox_from_mask,
    compute_image_stats,
    decode_bitmask,
    decode_depth_map,
    encode_bitmask,
    load_lance_batch,
    segmentation_stub,
)
from pdebug.utils.gpu_memory import gpu_memory_tic, gpu_memory_toc

import cv2
import numpy as np
import pyarrow as pa
import typer

try:  # torch is optional; required for full inference
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow optional for encoding/decoding
    Image = None  # type: ignore[assignment]


app = typer.Typer(
    help="Execute Cortexia pipeline stages over a Lance dataset."
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_reader_kwargs(
    *,
    timestamp_col: Optional[str],
    video_id_col: Optional[str],
    frame_num_col: Optional[str],
    row_limit: Optional[int],
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if timestamp_col:
        kwargs["timestamp_col"] = timestamp_col
    if video_id_col:
        kwargs["video_id_col"] = video_id_col
    if frame_num_col:
        kwargs["frame_num_col"] = frame_num_col
    if row_limit:
        kwargs["row_limit"] = int(row_limit)
    return kwargs


def _dicts_to_struct_array(payloads: Sequence[Dict[str, object]]) -> pa.Array:
    if not payloads:
        return pa.array([], type=pa.null())
    return pa.array(payloads)


def _append_columns(table: pa.Table, columns: Dict[str, pa.Array]) -> pa.Table:
    updated = table
    for name, array in columns.items():
        if name in updated.column_names:
            idx = updated.column_names.index(name)
            updated = updated.set_column(idx, name, array)
        else:
            updated = updated.append_column(name, array)
    return updated


_VIS_GRID_ROWS = 3
_VIS_GRID_COLS = 2


def _write_lance_dataset(table: pa.Table, output_path: Path) -> Path:
    import lance

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    lance.write_dataset(table, str(output_path))
    return output_path


def _preview_rows(table: pa.Table, limit: int = 3) -> None:
    rows = min(limit, len(table))
    typer.echo(f"Previewing first {rows} rows:")
    for idx in range(rows):
        row = table.slice(idx, 1)
        summary = {}
        for column in (
            "cortexia_caption",
            "cortexia_tags",
            "cortexia_detection",
            "cortexia_segmentation",
            "cortexia_depth",
        ):
            if column in row.column_names:
                cell = row[column][0]
                summary[column] = (
                    cell.as_py() if hasattr(cell, "as_py") else cell
                )
        if (
            "cortexia_segmentation" in summary
            and "mask" in summary["cortexia_segmentation"]
        ):
            summary["cortexia_segmentation"]["mask"] = str(
                summary["cortexia_segmentation"]["mask"]
            )[-100:]
        if "cortexia_depth" in summary and "map" in summary["cortexia_depth"]:
            summary["cortexia_depth"]["map"] = str(
                summary["cortexia_depth"]["map"]
            )[-100:]
        typer.echo(f"\n  Row {idx}: {summary}")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_caption(
    images: Sequence[np.ndarray],
    *,
    unittest: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for idx, image in enumerate(images):
        results.append(
            moondream_module._moondream_infer(
                image, index=idx, unittest=unittest
            )
        )
    return results


def _run_listing(
    images: Sequence[np.ndarray],
    *,
    unittest: bool,
    prompt: str,
) -> List[Dict[str, object]]:
    if unittest:
        results: List[Dict[str, object]] = []
        for idx, image in enumerate(images):
            stats = compute_image_stats(image)
            results.append(qwen2_5_vl._dummy_tags(stats, idx))
        return results
    return qwen2_5_vl._real_qwen_dataset_infer(images, prompt=prompt)


def _prompt_from_tags(tags: Iterable[str]) -> str:
    cleaned = [tag.strip() for tag in tags if tag and isinstance(tag, str)]
    if not cleaned:
        return "all objects."
    joined = ", ".join(dict.fromkeys(cleaned))
    return f"Detect: {joined}"


def _valid_bbox(box: Sequence[float], width: int, height: int) -> bool:
    if len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if x2 < 0 or y2 < 0:
        return False
    if x1 > width or y1 > height:
        return False
    return True


def _run_detection(
    images: Sequence[np.ndarray],
    listing_results: Sequence[Dict[str, object]],
    *,
    unittest: bool,
) -> List[Dict[str, object]]:
    prompts = []
    for result in listing_results:
        tags = result.get("tags") if isinstance(result, dict) else None
        prompts.append(_prompt_from_tags(tags or []))

    if not unittest:
        # Ensure settings dict is reused so we can update prompt per frame.
        _, _, settings = groundingdino_module._load_groundingdino_model()

    detections: List[Dict[str, object]] = []
    for idx, image in enumerate(images):
        if not unittest:
            settings["prompt"] = prompts[idx]
        detections.append(
            groundingdino_module._grounding_infer(
                image, index=idx, unittest=unittest
            )
        )
    return detections


def _run_segmentation(
    images: Sequence[np.ndarray],
    *,
    unittest: bool,
) -> List[Dict[str, object]]:

    results: List[Dict[str, object]] = []

    repo = semseg_module.prepare()
    model, infer_func, _ = semseg_module.init_model(repo, "big", "cuda:0")

    for image in images:
        result = infer_func(model, image)[0]
        results.append(
            {
                "mask": result.reshape(-1),
                "model": "internimage_DCNv4",
                "shape": result.shape,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def _append_text_panel(
    image: np.ndarray, title: str, lines: Sequence[str]
) -> np.ndarray:
    from pdebug.visp import (
        draw as vis_draw,  # local import to avoid heavy dependency when unused
    )

    merged: List[str] = [title]
    merged.extend([line for line in lines if line])
    return vis_draw.text(image, merged, target_width=image.shape[1])


def _render_original(image: np.ndarray) -> np.ndarray:
    return _append_text_panel(image, "[Original]", [])


def _render_caption(
    image: np.ndarray, payload: Optional[Dict[str, object]]
) -> np.ndarray:
    caption = None
    if isinstance(payload, dict):
        caption = payload.get("caption") or payload.get("raw_response")
    caption = caption or "No caption."
    return _append_text_panel(image, "[Caption]", [str(caption)])


def _render_tags(
    image: np.ndarray, payload: Optional[Dict[str, object]]
) -> np.ndarray:
    tags = []
    raw = None
    if isinstance(payload, dict):
        tags = payload.get("tags") or []
        raw = payload.get("raw_response")
    summary = ", ".join(tags) if tags else (raw or "No tags.")
    return _append_text_panel(image, "[Listing]", [summary])


def _render_detection(
    image: np.ndarray, payload: Optional[Dict[str, object]]
) -> np.ndarray:
    from pdebug.visp import draw as vis_draw

    detections = []
    if isinstance(payload, dict):
        detections = payload.get("detections") or []

    height, width = image.shape[:2]
    boxes: List[Sequence[float]] = []
    labels: List[str] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        box = det.get("bbox")
        if not isinstance(box, (list, tuple)):
            continue
        if not _valid_bbox(box, width, height):
            continue
        score = det.get("score")
        label = det.get("label") or "object"
        if isinstance(score, (float, int)):
            labels.append(f"{label}:{float(score):.2f}")
        else:
            labels.append(str(label))
        boxes.append(box)

    rendered = image.copy()
    if boxes:
        np_boxes = np.asarray(boxes, dtype=np.float32)
        rendered = vis_draw.boxes(rendered, np_boxes)
        rendered = vis_draw.flag(rendered, labels, np_boxes)
        summary = f"{len(boxes)} detections"
    else:
        summary = "No detections"
    return _append_text_panel(rendered, "[Detection]", [summary])


def _render_segmentation(
    image: np.ndarray, payload: Optional[Dict[str, object]]
) -> np.ndarray:
    from pdebug.visp import draw as vis_draw

    mask = np.asarray(payload["mask"]).reshape(payload["shape"])
    rendered = vis_draw.semseg(
        mask,
        image=image,
        colors=semseg_module.COLOR_PALETTE,
        classes=semseg_module.ADE_CLASSES,
    )
    summary = ""
    return _append_text_panel(rendered, "[Segmentation]", [summary])


def _render_depth(
    image: np.ndarray, payload: Optional[Dict[str, object]]
) -> np.ndarray:
    from pdebug.visp import draw as vis_draw

    depth_data: Dict[str, object] = {}
    if isinstance(payload, dict):
        depth_data = payload
    lines = []
    for key in ("mean_depth", "min_depth", "max_depth"):
        value = depth_data.get(key)
        if isinstance(value, (float, int)):
            lines.append(f"{key}: {float(value):.3f}")

    depth_map = decode_depth_map(depth_data.get("map")) if depth_data else None
    if depth_map is None:
        return _append_text_panel(image, "[Depth]", lines or ["No depth map"])

    height, width = image.shape[:2]
    if depth_map.shape != (height, width):
        if cv2 is not None:
            depth_map = cv2.resize(
                depth_map, (width, height), interpolation=cv2.INTER_LINEAR
            )
        elif Image is not None:
            resized = Image.fromarray(depth_map.astype(np.float32)).resize(
                (width, height), resample=Image.BILINEAR
            )
            depth_map = np.array(resized, dtype=np.float32)
        else:
            return _append_text_panel(
                image, "[Depth]", lines or ["Depth map requires OpenCV/Pillow"]
            )

    finite_mask = np.isfinite(depth_map)
    if not finite_mask.any():
        valid = np.zeros_like(depth_map, dtype=np.float32)
    else:
        valid = (
            depth_map[finite_mask].astype(np.float32).reshape(depth_map.shape)
        )
        # remove unvalid value
        valid[valid > 100] = 0

    overlay = vis_draw.depth(valid)
    return _append_text_panel(
        overlay, "[Depth]", lines or ["No depth statistics"]
    )


def _combine_visualizations(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No visualization images provided.")
    rows, cols = _VIS_GRID_ROWS, _VIS_GRID_COLS
    cell_height = max(img.shape[0] for img in images)
    cell_width = max(img.shape[1] for img in images)

    def _resize_to_cell(img: np.ndarray) -> np.ndarray:
        img_uint8 = _ensure_uint8(img)
        height, width = img_uint8.shape[:2]
        scale = min(cell_width / width, cell_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if (new_width, new_height) != (width, height):
            if cv2 is not None:
                resized = cv2.resize(
                    img_uint8,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                    if scale < 1.0
                    else cv2.INTER_LINEAR,
                )
            elif Image is not None:
                resized = Image.fromarray(img_uint8).resize(
                    (new_width, new_height),
                    resample=Image.BILINEAR if scale >= 1.0 else Image.BICUBIC,
                )
                resized = np.asarray(resized, dtype=np.uint8)
            else:
                resized = img_uint8
        else:
            resized = img_uint8
        canvas = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        y_off = (cell_height - resized.shape[0]) // 2
        x_off = (cell_width - resized.shape[1]) // 2
        canvas[
            y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]
        ] = resized
        return canvas

    max_cells = rows * cols
    prepared = [_resize_to_cell(img) for img in images[:max_cells]]
    while len(prepared) < max_cells:
        prepared.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))

    canvas = np.zeros(
        (rows * cell_height, cols * cell_width, 3), dtype=np.uint8
    )
    for idx, img in enumerate(prepared):
        row = idx // cols
        col = idx % cols
        y_start = row * cell_height
        x_start = col * cell_width
        canvas[
            y_start : y_start + cell_height, x_start : x_start + cell_width
        ] = img
    return canvas


def _render_error(
    image: np.ndarray, title: str, error: Exception
) -> np.ndarray:
    message = str(error)
    return _append_text_panel(
        image, title, [f"Visualization error: {message}"]
    )


def _visualize_models(
    dataset_path: Path,
    *,
    image_col: str,
    reader_kwargs: Dict[str, object],
    output_dir: Path,
) -> None:

    batch = load_lance_batch(
        dataset_path,
        image_col=image_col,
        reader_kwargs=reader_kwargs,
    )
    if not batch.images:
        raise RuntimeError("No images found for visualization.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(batch.metadata):
        image = entry.get("image")
        if not isinstance(image, np.ndarray):
            raise TypeError("Lance entry image is missing or invalid.")
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
                f"Visualization failed on row {index}: {exc}",
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
        frame_name = entry.get("image_name") or f"frame_{index:06d}"
        safe_name = Path(str(frame_name)).stem
        output_path = output_dir / f"{index:04d}_{safe_name}.png"
        bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(
            str(output_path), bgr
        ):  # pragma: no cover - cv2 returns bool
            raise RuntimeError(
                f"Failed to write visualization to {output_path}"
            )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_dataset: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
        help="Path to the source Lance dataset.",
    ),
    output_dataset: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination Lance dataset path. Defaults to <input>_annotated.lance.",
    ),
    vis_output: Optional[Path] = typer.Option(
        None,
        "--vis-output",
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
    unittest: bool = typer.Option(
        False,
        "--unittest/--no-unittest",
        help="Enable deterministic stub inference instead of running full models.",
    ),
    qwen_prompt: Optional[str] = typer.Option(
        None,
        "--qwen-prompt",
        help="Custom user prompt passed to Qwen2.5-VL.",
    ),
) -> None:
    dataset_path = input_dataset.resolve()
    if output_dataset is None:
        output_dataset = dataset_path.with_name(
            f"{dataset_path.stem}_annotated.lance"
        )
    output_dataset = output_dataset.resolve()

    if dataset_path == output_dataset:
        raise typer.BadParameter(
            "Output dataset must differ from input dataset."
        )

    reader_kwargs = _build_reader_kwargs(
        timestamp_col=timestamp_col,
        video_id_col=video_id_col,
        frame_num_col=frame_num_col,
        row_limit=row_limit,
    )

    typer.echo(f"Loading Lance dataset from {dataset_path} ...")
    batch = load_lance_batch(
        dataset_path,
        image_col=image_col,
        reader_kwargs=reader_kwargs,
    )
    if not batch.images:
        raise RuntimeError(f"No images found in Lance dataset {dataset_path}")
    typer.echo(f"Processing {len(batch.images)} rows.")

    prompt = qwen_prompt or qwen2_5_vl.DEFAULT_TEXT

    typer.echo("Running Moondream caption inference ...")
    caption_before = gpu_memory_tic()
    caption_results = _run_caption(batch.images, unittest=unittest)
    gpu_memory_toc(
        "Moondream caption inference",
        caption_before,
        (getattr(moondream_module, "_load_moondream_model", None),),
    )

    typer.echo("Running Qwen2.5-VL listing inference ...")
    listing_before = gpu_memory_tic()
    listing_results = _run_listing(
        batch.images, unittest=unittest, prompt=prompt
    )
    gpu_memory_toc(
        "Qwen2.5-VL listing inference",
        listing_before,
        (getattr(qwen2_5_vl, "_load_qwen_model", None),),
    )

    typer.echo("Running GroundingDINO detection with listing prompts ...")
    detection_before = gpu_memory_tic()
    detection_results = _run_detection(
        batch.images, listing_results, unittest=unittest
    )
    gpu_memory_toc(
        "GroundingDINO detection",
        detection_before,
        (getattr(groundingdino_module, "_load_groundingdino_model", None),),
    )

    typer.echo("Running InternImage seg ...")
    segmentation_before = gpu_memory_tic()
    segmentation_results = _run_segmentation(batch.images, unittest=unittest)
    gpu_memory_toc(
        "Internimage segmentation",
        segmentation_before,
        (getattr(semseg_module, "_load_internimage_model", None),),
    )

    typer.echo("Running ML-Depth-Pro depth estimation ...")
    depth_before = gpu_memory_tic()
    depth_results: List[Dict[str, object]] = []
    for image in batch.images:
        depth_results.append(ml_depth_pro_module._depth_infer(image, unittest=unittest))  # type: ignore[attr-defined]
    gpu_memory_toc(
        "ML-Depth-Pro depth estimation",
        depth_before,
        (getattr(ml_depth_pro_module, "_load_depth_pro_model", None),),
    )

    column_map = {
        "cortexia_caption": _dicts_to_struct_array(caption_results),
        "cortexia_tags": _dicts_to_struct_array(listing_results),
        "cortexia_detection": _dicts_to_struct_array(detection_results),
        "cortexia_segmentation": _dicts_to_struct_array(segmentation_results),
        "cortexia_depth": _dicts_to_struct_array(depth_results),
    }

    annotated = _append_columns(batch.table, column_map)
    typer.echo(f"Writing annotated dataset to {output_dataset} ...")
    _write_lance_dataset(annotated, output_dataset)
    typer.echo(f"Annotated dataset saved to {output_dataset}")

    if vis_output is not None:
        _preview_rows(annotated, limit=3)

        typer.echo(f"Generating visualizations in {vis_output} ...")
        _visualize_models(
            output_dataset,
            image_col=image_col,
            reader_kwargs=reader_kwargs,
            output_dir=vis_output,
        )
        typer.echo(f"Visualization collages saved to: {vis_output}")


if __name__ == "__main__":
    app()
