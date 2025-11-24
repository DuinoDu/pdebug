#!/usr/bin/env python3
"""Run Cortexia-style pipelines over a Lance dataset using pdebug OTN nodes.

Install env steps:
    pip3 install -e ".[all]"
    FORCE_INSTALL_TORCH=1 bash $INSTALL/torch.sh 2.7.0    # moondream_node required torch = 2.7.0 
    bash $INSTALL/flash_attn.sh     # qwen2_5_vl
    bash $INSTALL/DCNv4.sh          # internimage_semseg, including mmcv_full==1.5.0

"""

from __future__ import annotations
import gc
import io
import itertools
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from pdebug.otn.infer import groundingdino_node as groundingdino_module
from pdebug.otn.infer import internimage_semseg as semseg_module
from pdebug.otn.infer import ml_depth_pro_node as ml_depth_pro_module
from pdebug.otn.infer import moondream_node as moondream_module
from pdebug.otn.infer import qwen2_5_vl
from pdebug.otn.infer.lance_utils import (
    LanceBatch,
    compute_image_stats,
    decode_bitmask,
    decode_depth_map,
    encode_depth_map_uint16,
    decode_depth_map_uint16,
    load_lance_batch,
)
from pdebug.utils.cache import ResultCache
from pdebug.utils.gpu_memory import gpu_memory_tic, gpu_memory_toc

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import typer
from loguru import logger

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


def _ensure_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"Unsupported image storage type: {type(value)}")


def _decode_lance_image(buffer: bytes, *, to_rgb: bool = True) -> np.ndarray:
    if buffer is None:
        raise ValueError("Encountered empty image buffer when decoding Lance data.")
    if Image is not None:
        with Image.open(io.BytesIO(buffer)) as handle:
            frame = np.asarray(handle.convert("RGB"))
    else:
        if cv2 is None:
            raise RuntimeError(
                "Decoding Lance images requires pillow or opencv-python to be installed."
            )
        frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image buffer via OpenCV.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not to_rgb and frame.ndim == 3:
        if cv2 is not None:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame[..., ::-1]
    return frame


def _timestamp_to_seconds(
    timestamp_value: object, frame_number: int
) -> Optional[float]:
    if timestamp_value is None:
        return frame_number / 30.0
    if isinstance(timestamp_value, float):
        return timestamp_value
    if isinstance(timestamp_value, int):
        if abs(timestamp_value) > 10_000_000_000:
            return timestamp_value / 1_000_000_000.0
        return float(timestamp_value)
    return None


def _cuda_warmup() -> None:
    if torch is None:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        pass


def _table_to_lance_batch(
    table: pa.Table,
    *,
    image_col: str,
    timestamp_col: Optional[str],
    video_id_col: Optional[str],
    frame_num_col: Optional[str],
    index_offset: int,
    trigger: Optional[str] = None,
) -> LanceBatch:
    images: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    if trigger:
        table = table.filter(pc.equal(table['trigger'], pa.scalar(trigger)))
        print(f"Filtered table to {len(table)} rows with trigger: {trigger}")

    for row_idx in range(len(table)):
        row = table.slice(row_idx, 1)
        entry: Dict[str, object] = {}
        for name in row.column_names:
            cell = row[name][0]
            value = cell.as_py() if hasattr(cell, "as_py") else cell
            entry[name] = value

        if image_col not in entry:
            raise KeyError(f"Column '{image_col}' missing from Lance batch slice.")

        buffer = _ensure_bytes(entry.pop(image_col))
        image = _decode_lance_image(buffer)
        entry["image"] = image
        entry["image_height"], entry["image_width"] = image.shape[:2]

        frame_index = index_offset + row_idx
        entry["frame_index"] = frame_index

        image_name = entry.get("image_name")
        if not image_name:
            entry["image_name"] = f"lance_{frame_index:06d}.png"

        frame_number = entry.get(frame_num_col) if frame_num_col else None
        if frame_number is None:
            frame_number = frame_index
        entry["frame_number_resolved"] = int(frame_number)

        source_video = entry.get(video_id_col) if video_id_col else None
        if source_video is None:
            source_video = entry.get("source_video_id_resolved") or "lance_vita"
        entry["source_video_id_resolved"] = str(source_video)

        timestamp_raw = entry.get(timestamp_col) if timestamp_col else entry.get(
            "timestamp_raw"
        )
        entry["timestamp_raw"] = timestamp_raw
        entry["timestamp_seconds"] = _timestamp_to_seconds(
            timestamp_raw, entry["frame_number_resolved"]
        )

        metadata.append(entry)
        images.append(image)

    return LanceBatch(table=table, images=images, metadata=metadata)


def _iter_lance_batches(
    dataset_path: Path,
    *,
    image_col: str,
    reader_kwargs: Dict[str, object],
    batch_size: Optional[int],
    timestamp_col: Optional[str],
    video_id_col: Optional[str],
    frame_num_col: Optional[str],
    trigger: Optional[str] = None,
) -> Iterator[LanceBatch]:
    import lance

    scanner_kwargs: Dict[str, object] = {}
    columns = reader_kwargs.get("columns")
    if columns:
        scanner_kwargs["columns"] = columns
    if batch_size and batch_size > 0:
        scanner_kwargs["batch_size"] = int(batch_size)

    ds = lance.dataset(str(dataset_path))
    scanner = ds.scanner(**scanner_kwargs)

    processed = 0
    row_limit = reader_kwargs.get("row_limit")
    limit_value = int(row_limit) if row_limit is not None else None

    for record_batch in scanner.to_batches():
        table = pa.Table.from_batches([record_batch]).combine_chunks()
        if limit_value is not None:
            rows_remaining = limit_value - processed
            if rows_remaining <= 0:
                break
            if len(table) > rows_remaining:
                table = table.slice(0, rows_remaining)

        if len(table) == 0:
            continue

        batch = _table_to_lance_batch(
            table,
            image_col=image_col,
            timestamp_col=timestamp_col,
            video_id_col=video_id_col,
            frame_num_col=frame_num_col,
            index_offset=processed,
            trigger=trigger,
        )

        yield batch
        processed += len(batch.images)

        if limit_value is not None and processed >= limit_value:
            break


def _slice_lance_batch(batch: LanceBatch, start: int) -> LanceBatch:
    if start <= 0:
        return batch
    total = len(batch.images)
    if start >= total:
        empty_table = batch.table.slice(total, 0)
        return LanceBatch(table=empty_table, images=[], metadata=[])
    images = list(batch.images[start:])
    metadata = list(batch.metadata[start:])
    table = batch.table.slice(start, len(images))
    return LanceBatch(table=table, images=images, metadata=metadata)


def _resume_batches(
    batches: Iterator[LanceBatch],
    *,
    resume_count: int,
    stage: str,
) -> Iterator[LanceBatch]:
    if resume_count <= 0:
        yield from batches
        return

    skipped = 0
    for batch_index, batch in enumerate(batches, start=1):
        span = len(batch.images)
        if span == 0:
            del batch
            gc.collect()
            continue
        if skipped + span <= resume_count:
            skipped += span
            logger.info(
                f"[{stage}] Resume skipping batch {batch_index} ({span} rows)."
            )
            del batch
            gc.collect()
            continue
        if skipped < resume_count:
            offset = resume_count - skipped
            logger.info(
                f"[{stage}] Resume skipping first {offset} rows from batch {batch_index}."
            )
            batch = _slice_lance_batch(batch, offset)
            skipped = resume_count
            if not batch.images:
                del batch
                gc.collect()
                continue
        yield batch
        skipped += len(batch.images)


class LanceDatasetWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._dataset = None
        self._initialized = False

    def write_batch(self, table: pa.Table) -> None:
        if len(table) == 0:
            return

        import lance

        if not self._initialized:
            if self.output_path.exists():
                shutil.rmtree(self.output_path)
            lance.write_dataset(table, str(self.output_path))
            self._dataset = lance.dataset(str(self.output_path))
            self._initialized = True
            return

        if self._dataset is None:
            self._dataset = lance.dataset(str(self.output_path))
        self._dataset.insert(table, mode="append")

_VIS_GRID_ROWS = 3
_VIS_GRID_COLS = 2


def _preview_rows(table: pa.Table, limit: int = 3) -> None:
    rows = min(limit, len(table))
    logger.info(f"Previewing first {rows} rows:")
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
        logger.info(f"\n  Row {idx}: {summary}")


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
    model_bundle: Optional[Tuple[object, object, object]] = None
    if not unittest:
        model_bundle = semseg_module._load_internimage_model("big", "cuda:0")

    for image in images:
        if unittest:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[: height // 2 or 1, : width // 2 or 1] = 1
            results.append(
                {
                    "mask": mask.reshape(-1),
                    "model": "internimage_DCNv4_stub",
                    "shape": mask.shape,
                }
            )
            continue

        if model_bundle is None:
            raise RuntimeError("Failed to load InternImage segmentation model.")
        model, infer_func, _ = model_bundle
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
    
    if "float16" in depth_data["map"]["encoding"]:
        depth_map = decode_depth_map(depth_data.get("map")) if depth_data else None
    elif "uint16_png" in depth_data["map"]["encoding"]:
        depth_map = decode_depth_map_uint16(depth_data.get("map")) if depth_data else None
    else:
        raise NotImplementedError
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
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Number of rows to materialize per batch.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Reuse cached stage outputs to continue a previous run.",
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
    trigger: Optional[str] = typer.Option(
        None,
        "--trigger",
        help="Trigger to filter the dataset.",
    ),
    tasks: Optional[List[str]] = typer.Option(
        None,
        "--task",
        "-t",
        help=(
            "Pipeline stages to execute. Repeat to select multiple from "
            "caption, listing, detection, segmentation, depth. Defaults to "
            "all stages. Example: --task segmentation to only run "
            "segmentation or -t caption -t segmentation to run those two "
            "stages only."
        ),
    ),
) -> None:
    available_stages: Tuple[str, ...] = (
        "caption",
        "listing",
        "detection",
        "segmentation",
        "depth",
    )

    if tasks:
        normalized_tasks = [task.lower() for task in tasks]
        invalid = [task for task in normalized_tasks if task not in available_stages]
        if invalid:
            raise typer.BadParameter(
                f"Unsupported task(s) requested: {invalid}. "
                f"Choose from {', '.join(available_stages)}."
            )

        requested_tasks = set(normalized_tasks)
        if "detection" in requested_tasks and "listing" not in requested_tasks:
            logger.info(
                "Including listing task because detection depends on listing prompts."
            )
            requested_tasks.add("listing")

        stages_to_run: Tuple[str, ...] = tuple(
            stage for stage in available_stages if stage in requested_tasks
        )
    else:
        stages_to_run = available_stages

    if not stages_to_run:
        raise typer.BadParameter("At least one task must be selected.")

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

    logger.info(f"Scanning Lance dataset from {dataset_path} ...")
    import lance

    ds = lance.dataset(str(dataset_path))
    total_rows = ds.count_rows()
    if row_limit is not None:
        total_rows = min(int(row_limit), total_rows)
    if total_rows <= 0:
        raise RuntimeError(
            f"No images found in Lance dataset {dataset_path}"
        )
    del ds

    effective_batch_size = (
        int(batch_size) if batch_size is not None else total_rows
    )
    if effective_batch_size <= 0:
        effective_batch_size = total_rows
    effective_batch_size = max(1, min(effective_batch_size, total_rows))

    logger.info(
        f"Processing {total_rows} rows in batches of {effective_batch_size}."
    )
    def iter_batches() -> Iterator[LanceBatch]:
        return _iter_lance_batches(
            dataset_path,
            image_col=image_col,
            reader_kwargs=reader_kwargs,
            batch_size=effective_batch_size,
            timestamp_col=timestamp_col,
            video_id_col=video_id_col,
            frame_num_col=frame_num_col,
            trigger=trigger,
        )

    _cuda_warmup()

    cache_root: Optional[Path] = None
    if resume:
        resume_root = (
            output_dataset.parent
            / ".use_with_lance_cache"
            / dataset_path.name
        )
        if row_limit is not None:
            resume_root = resume_root / f"rows_{int(row_limit)}"
        if unittest:
            resume_root = resume_root / "unittest"
        cache_root = resume_root.resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Resume cache directory: {cache_root}")

    def _make_cache(stage: str) -> ResultCache:
        if cache_root is None:
            return ResultCache(stage)
        return ResultCache(
            stage,
            persist_path=cache_root / stage,
            resume=True,
        )

    caches: List[ResultCache] = []

    caption_cache = _make_cache("caption")
    listing_cache = _make_cache("listing")
    detection_cache = _make_cache("detection")
    segmentation_cache = _make_cache("segmentation")
    depth_cache = _make_cache("depth")

    caches.extend(
        [
            caption_cache,
            listing_cache,
            detection_cache,
            segmentation_cache,
            depth_cache,
        ]
    )

    if resume:
        logger.info(
            f"Loaded cache counts - caption:{caption_cache.count} "
            f"listing:{listing_cache.count} detection:{detection_cache.count} "
            f"segmentation:{segmentation_cache.count} depth:{depth_cache.count}"
        )

    preview_tables: List[pa.Table] = []
    preview_remaining = 3

    run_success = False

    try:
        # Stage 1: Caption
        if "caption" in stages_to_run:
            logger.info(f"Running Moondream caption inference ...")
            caption_before = gpu_memory_tic()
            caption_resume = min(caption_cache.count, total_rows) if resume else 0
            if resume and caption_cache.count > total_rows:
                logger.warning(
                    "Caption cache contains %d rows but dataset has %d; "
                    "only the first %d rows will be reused.",
                    caption_cache.count,
                    total_rows,
                    caption_resume,
                )
            if resume and caption_resume >= total_rows:
                logger.info(
                    f"Caption cache already covers all rows; skipping stage."
                )
            processed = caption_resume
            for batch in _resume_batches(
                iter_batches(), resume_count=caption_resume, stage="Moondream"
            ):
                logger.info(f"[Moondream] {processed} / {total_rows}")
                if not batch.images:
                    del batch
                    gc.collect()
                    continue
                caption_cache.append_many(
                    _run_caption(batch.images, unittest=unittest)
                )
                processed += len(batch.images)
                del batch
                gc.collect()
            caption_cache.finalize()
            if processed != total_rows:
                raise RuntimeError(
                    "Caption stage processed an unexpected number of rows."
                )
            gpu_memory_toc(
                "Moondream caption inference",
                caption_before,
                (getattr(moondream_module, "_load_moondream_model", None),),
            )
        else:
            logger.info("Skipping caption stage (not requested).")

        # Stage 2: Listing
        if "listing" in stages_to_run:
            logger.info(f"Running Qwen2.5-VL listing inference ...")
            prompt = qwen_prompt or qwen2_5_vl.DEFAULT_TEXT
            listing_before = gpu_memory_tic()
            listing_resume = min(listing_cache.count, total_rows) if resume else 0
            if resume and listing_cache.count > total_rows:
                logger.warning(
                    "Listing cache contains %d rows but dataset has %d; "
                    "only the first %d rows will be reused.",
                    listing_cache.count,
                    total_rows,
                    listing_resume,
                )
            if resume and listing_resume >= total_rows:
                logger.info(
                    f"Listing cache already covers all rows; skipping stage."
                )
            processed = listing_resume
            for batch in _resume_batches(
                iter_batches(), resume_count=listing_resume, stage="Qwen2.5-VL"
            ):
                logger.info(f"[Qwen2.5-VL] {processed} / {total_rows}")
                if not batch.images:
                    del batch
                    gc.collect()
                    continue
                listing_cache.append_many(
                    _run_listing(batch.images, unittest=unittest, prompt=prompt)
                )
                processed += len(batch.images)
                del batch
                gc.collect()
            listing_cache.finalize()
            if processed != total_rows:
                raise RuntimeError(
                    "Listing stage processed an unexpected number of rows."
                )
            gpu_memory_toc(
                "Qwen2.5-VL listing inference",
                listing_before,
                (getattr(qwen2_5_vl, "_load_qwen_model", None),),
            )
        else:
            logger.info("Skipping listing stage (not requested).")

        # Stage 3: Detection (depends on listing results)
        if "detection" in stages_to_run:
            if "listing" not in stages_to_run and listing_cache.count < total_rows:
                raise RuntimeError(
                    "Detection stage requires listing results; rerun with the "
                    "listing task or ensure listing cache is complete."
                )
            logger.info(
                f"Running GroundingDINO detection with listing prompts ..."
            )
            detection_before = gpu_memory_tic()
            listing_iter_for_detection = listing_cache.iter_results()
            detection_resume = min(detection_cache.count, total_rows) if resume else 0
            if resume and detection_cache.count > total_rows:
                logger.warning(
                    "Detection cache contains %d rows but dataset has %d; "
                    "only the first %d rows will be reused.",
                    detection_cache.count,
                    total_rows,
                    detection_resume,
                )
            if detection_resume:
                skipped_prompts = 0
                while skipped_prompts < detection_resume:
                    entry = next(listing_iter_for_detection, None)
                    if entry is None:
                        raise RuntimeError(
                            "Detection resume mismatch: not enough listing prompts cached."
                        )
                    skipped_prompts += 1
            if resume and detection_resume >= total_rows:
                logger.info(
                    f"Detection cache already covers all rows; skipping stage."
                )
            processed = detection_resume
            for batch in _resume_batches(
                iter_batches(),
                resume_count=detection_resume,
                stage="GroundingDINO",
            ):
                logger.info(f"[GroundingDINO] {processed} / {total_rows}")
                if not batch.images:
                    del batch
                    gc.collect()
                    continue
                span = len(batch.images)
                listing_slice = list(
                    itertools.islice(listing_iter_for_detection, span)
                )
                if len(listing_slice) != span:
                    raise RuntimeError(
                        "Detection stage could not align listing results with images."
                    )
                detection_cache.append_many(
                    _run_detection(batch.images, listing_slice, unittest=unittest)
                )
                processed += span
                del batch
                gc.collect()
            detection_cache.finalize()
            if processed != total_rows:
                raise RuntimeError(
                    "Detection stage processed an unexpected number of rows."
                )
            gpu_memory_toc(
                "GroundingDINO detection",
                detection_before,
                (getattr(groundingdino_module, "_load_groundingdino_model", None),),
            )
        else:
            logger.info("Skipping detection stage (not requested).")

        # Stage 4: Segmentation
        if "segmentation" in stages_to_run:
            logger.info(f"Running InternImage segmentation ...")
            segmentation_before = gpu_memory_tic()
            segmentation_resume = (
                min(segmentation_cache.count, total_rows) if resume else 0
            )
            if resume and segmentation_cache.count > total_rows:
                logger.warning(
                    "Segmentation cache contains %d rows but dataset has %d; "
                    "only the first %d rows will be reused.",
                    segmentation_cache.count,
                    total_rows,
                    segmentation_resume,
                )
            if resume and segmentation_resume >= total_rows:
                logger.info(
                    f"Segmentation cache already covers all rows; skipping stage."
                )
            processed = segmentation_resume
            for batch in _resume_batches(
                iter_batches(), resume_count=segmentation_resume, stage="Internimage"
            ):
                logger.info(f"[Internimage] {processed} / {total_rows}")
                if not batch.images:
                    del batch
                    gc.collect()
                    continue
                segmentation_cache.append_many(
                    _run_segmentation(batch.images, unittest=unittest)
                )
                processed += len(batch.images)
                del batch
                gc.collect()
            segmentation_cache.finalize()
            if processed != total_rows:
                raise RuntimeError(
                    "Segmentation stage processed an unexpected number of rows."
                )
            gpu_memory_toc(
                "Internimage segmentation",
                segmentation_before,
                (getattr(semseg_module, "_load_internimage_model", None),),
            )
        else:
            logger.info("Skipping segmentation stage (not requested).")

        # Stage 5: Depth
        if "depth" in stages_to_run:
            logger.info(f"Running ML-Depth-Pro depth estimation ...")
            depth_before = gpu_memory_tic()
            depth_resume = min(depth_cache.count, total_rows) if resume else 0
            if resume and depth_cache.count > total_rows:
                logger.warning(
                    "Depth cache contains %d rows but dataset has %d; "
                    "only the first %d rows will be reused.",
                    depth_cache.count,
                    total_rows,
                    depth_resume,
                )
            if resume and depth_resume >= total_rows:
                logger.info(
                    f"Depth cache already covers all rows; skipping stage."
                )
            processed = depth_resume
            for batch in _resume_batches(
                iter_batches(), resume_count=depth_resume, stage="ML-Depth-Pro"
            ):
                logger.info(f"[ML-Depth-Pro] {processed} / {total_rows}")
                if not batch.images:
                    del batch
                    gc.collect()
                    continue
                for image in batch.images:
                    depth_cache.append(
                        ml_depth_pro_module._depth_infer(image, unittest=unittest)  # type: ignore[attr-defined]
                    )
                processed += len(batch.images)
                del batch
                gc.collect()
            depth_cache.finalize()
            if processed != total_rows:
                raise RuntimeError(
                    "Depth stage processed an unexpected number of rows."
                )
            gpu_memory_toc(
                "ML-Depth-Pro depth estimation",
                depth_before,
                (getattr(ml_depth_pro_module, "_load_depth_pro_model", None),),
            )
        else:
            logger.info("Skipping depth stage (not requested).")

        # Final write pass aggregating all stage outputs
        writer = LanceDatasetWriter(output_dataset)
        processed = 0

        caption_iter = caption_cache.iter_results() if "caption" in stages_to_run else None
        listing_iter = listing_cache.iter_results() if "listing" in stages_to_run else None
        detection_iter = (
            detection_cache.iter_results() if "detection" in stages_to_run else None
        )
        segmentation_iter = (
            segmentation_cache.iter_results()
            if "segmentation" in stages_to_run
            else None
        )
        depth_iter = depth_cache.iter_results() if "depth" in stages_to_run else None

        for batch_index, batch in enumerate(iter_batches(), start=1):
            table = batch.table
            span = len(table)
            if span == 0:
                del batch
                continue

            column_map = {}

            if caption_iter is not None:
                caption_slice = list(itertools.islice(caption_iter, span))
                if len(caption_slice) != span:
                    raise RuntimeError(
                        "Final write pass received mismatched caption chunk size."
                    )
                column_map["cortexia_caption"] = _dicts_to_struct_array(
                    caption_slice
                )

            listing_slice: List[dict] = []
            if listing_iter is not None:
                listing_slice = list(itertools.islice(listing_iter, span))
                if len(listing_slice) != span:
                    raise RuntimeError(
                        "Final write pass received mismatched listing chunk size."
                    )
                column_map["cortexia_tags"] = _dicts_to_struct_array(
                    listing_slice
                )

            if detection_iter is not None:
                detection_slice = list(itertools.islice(detection_iter, span))
                if len(detection_slice) != span:
                    raise RuntimeError(
                        "Final write pass received mismatched detection chunk size."
                    )
                column_map["cortexia_detection"] = _dicts_to_struct_array(
                    detection_slice
                )

            if segmentation_iter is not None:
                segmentation_slice = list(itertools.islice(segmentation_iter, span))
                if len(segmentation_slice) != span:
                    raise RuntimeError(
                        "Final write pass received mismatched segmentation chunk size."
                    )
                for item in segmentation_slice:
                    assert item["mask"].max() <= 255
                    item["mask"] = item["mask"].astype(np.uint8)

                column_map["cortexia_segmentation"] = _dicts_to_struct_array(
                    segmentation_slice
                )

            if depth_iter is not None:
                depth_slice = list(itertools.islice(depth_iter, span))
                if len(depth_slice) != span:
                    raise RuntimeError(
                        "Final write pass received mismatched depth chunk size."
                    )
                for item in depth_slice:
                    item.pop("processing_time_ms")
                    item.pop("mean_depth")
                    item.pop("model_name")
                    if "encoding" in item["map"]:
                        # convert to uint16 png compress
                        depth_map = decode_depth_map(item.get("map"))
                        item["map"] = encode_depth_map_uint16(depth_map)

                column_map["cortexia_depth"] = _dicts_to_struct_array(
                    depth_slice
                )

            annotated = _append_columns(table, column_map)
            writer.write_batch(annotated)

            if preview_remaining > 0:
                take = min(preview_remaining, len(annotated))
                if take > 0:
                    preview_tables.append(annotated.slice(0, take))
                    preview_remaining -= take

            processed += span
            logger.info(
                f"Completed batch {batch_index}: {processed}/{total_rows} rows written."
            )

            del batch
            gc.collect()

        if processed != total_rows:
            raise RuntimeError(
                "Final write pass processed an unexpected number of rows."
            )

        run_success = True

    finally:
        for cache in caches:
            cache.cleanup()

    if run_success and cache_root is not None and cache_root.exists():
        try:
            shutil.rmtree(cache_root)
            logger.info(f"Removed Lance cache directory: {cache_root}")
        except Exception as exc:
            logger.warning(
                "Unable to remove Lance cache directory %s: %s",
                cache_root,
                exc,
            )

    logger.info(f"Annotated dataset saved to {output_dataset}")

    preview_table: Optional[pa.Table] = None
    if preview_tables:
        preview_table = pa.concat_tables(preview_tables).combine_chunks()

    if vis_output is not None:
        if preview_table is not None:
            _preview_rows(preview_table, limit=3)

        logger.info(f"Generating visualizations in {vis_output} ...")
        _visualize_models(
            output_dataset,
            image_col=image_col,
            reader_kwargs=reader_kwargs,
            output_dir=vis_output,
        )
        logger.info(f"Visualization collages saved to: {vis_output}")


if __name__ == "__main__":
    app()
