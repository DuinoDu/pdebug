"""Utility helpers for Lance-backed inference nodes."""
from __future__ import annotations
import base64
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pdebug.piata import Input

import io
import cv2
import numpy as np
from PIL import Image


class LanceBatch:
    """Container for Lance dataset data."""

    __slots__ = ("table", "images", "metadata")

    def __init__(
        self,
        table: "pyarrow.Table",
        images: List[np.ndarray],
        metadata: List[Dict[str, object]],
    ) -> None:
        self.table = table
        self.images = images
        self.metadata = metadata


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from path/URL/array into RGB numpy array."""
    if isinstance(source, np.ndarray):
        if source.ndim != 3:
            raise ValueError("Expected 3-channel image array.")
        return source

    path = str(source)
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as handle:
            buffer = np.asarray(bytearray(handle.read()), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to decode image from {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_lance_batch(
    dataset_path: Union[str, Path],
    *,
    image_col: str = "camera_left",
    reader_kwargs: Optional[Dict[str, object]] = None,
) -> LanceBatch:
    """Load a Lance dataset into memory with decoded RGB frames."""
    reader_kwargs = reader_kwargs or {}
    reader = Input(
        str(dataset_path),
        name="lance_vita",
        image_col=image_col,
        **reader_kwargs,
    ).get_reader()

    images: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    for _ in range(len(reader)):
        entry = next(reader)
        metadata.append(entry)
        images.append(entry["image"])
    reader.reset()

    table = getattr(reader, "_table", None)
    if table is None:  # pragma: no cover - Lance reader always exposes _table
        import lance

        ds = lance.dataset(str(dataset_path))
        table = ds.to_table()

    return LanceBatch(table=table, images=images, metadata=metadata)


def _ensure_parent(path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(output: Union[str, Path], payload: Dict[str, object]) -> None:
    """Persist JSON payload with ensured parent directory."""
    target = _ensure_parent(output)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_lance_with_column(
    table: "pyarrow.Table",
    column_name: str,
    column_values: Sequence[object],
    output_path: Union[str, Path],
) -> Path:
    """Write Lance dataset with (re)placed column values."""
    import lance
    import pyarrow as pa

    if len(table) != len(column_values):
        raise ValueError(
            f"Table length {len(table)} differs from column size {len(column_values)}."
        )
    arr = pa.array(column_values)
    if column_name in table.column_names:
        idx = table.column_names.index(column_name)
        table = table.set_column(idx, column_name, arr)
    else:
        table = table.append_column(column_name, arr)

    output = _ensure_parent(output_path)
    if output.exists():
        shutil.rmtree(output)
    lance.write_dataset(table, str(output))
    return output


def compute_image_stats(image: np.ndarray) -> Dict[str, float]:
    """Return simple statistics for an image to craft deterministic outputs."""
    if image.ndim != 3:
        raise ValueError("Expected 3D image tensor.")
    height, width = image.shape[:2]
    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))
    norm_factor = float(
        np.linalg.norm(
            image[: min(4, height), : min(4, width), :].astype(np.float32)
        )
    )
    return {
        "height": float(height),
        "width": float(width),
        "mean": mean_intensity,
        "std": std_intensity,
        "norm": float(round(norm_factor, 4)),
    }


def deterministic_caption(stats: Dict[str, float], prefix: str) -> str:
    """Create a deterministic caption string from stats."""
    return (
        f"{prefix} {int(stats['width'])}x{int(stats['height'])} "
        f"mean={stats['mean']:.2f} std={stats['std']:.2f}"
    )


def scaled_bbox(stats: Dict[str, float], scale: float) -> List[float]:
    """Compute a simple pseudo bounding box based on image statistics."""
    cx = stats["width"] / 2.0
    cy = stats["height"] / 2.0
    half_w = max(10.0, stats["width"] * scale / 2.0)
    half_h = max(10.0, stats["height"] * scale / 3.0)
    xmin = max(0.0, cx - half_w)
    ymin = max(0.0, cy - half_h)
    xmax = min(stats["width"], cx + half_w)
    ymax = min(stats["height"], cy + half_h)
    return [
        float(round(xmin, 2)),
        float(round(ymin, 2)),
        float(round(xmax, 2)),
        float(round(ymax, 2)),
    ]


def encode_bitmask(mask: np.ndarray) -> Dict[str, object]:
    """Encode a binary mask into a compact base64 payload."""
    if mask.ndim != 2:
        raise ValueError("encode_bitmask expects a 2D mask.")
    height, width = mask.shape
    flattened = np.asarray(mask, dtype=np.uint8).reshape(-1)
    packed = np.packbits(flattened, axis=0)
    encoded = base64.b64encode(packed.tobytes()).decode("ascii")
    return {
        "encoding": "bitmask",
        "height": int(height),
        "width": int(width),
        "num_bits": int(height * width),
        "data": encoded,
    }


def encode_png_u8(mask: np.ndarray) -> bytes:
    """Encode a 2D uint8 mask as PNG bytes (single-channel)."""
    if mask.ndim != 2:
        raise ValueError("encode_png_u8 expects a 2D mask.")
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        ok, buf = cv2.imencode(".png", mask_u8)
        if not ok:
            raise ValueError("Failed to encode mask to PNG via OpenCV.")
        return buf.tobytes()
    if Image is not None:
        pil = Image.fromarray(mask_u8, mode="L")
        out = io.BytesIO()
        pil.save(out, format="PNG", optimize=True)
        return out.getvalue()
    raise RuntimeError("Encoding PNG requires opencv-python or pillow.")


def decode_png_u8(data: Union[bytes, bytearray, memoryview]) -> np.ndarray:
    """Decode PNG bytes into a 2D uint8 mask (single-channel)."""
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, (bytes,)):
        raise TypeError(f"decode_png_u8 expects bytes-like, got {type(data)}")
    if cv2 is not None:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode PNG bytes via OpenCV.")
        if img.ndim == 3:
            img = img[:, :, 0]
        return np.asarray(img, dtype=np.uint8)
    if Image is not None:
        pil = Image.open(io.BytesIO(data))
        return np.array(pil.convert("L"), dtype=np.uint8)
    raise RuntimeError("Decoding PNG requires opencv-python or pillow.")


def decode_bitmask(
    payload: Optional[Dict[str, object]]
) -> Optional[np.ndarray]:
    """Decode a mask previously produced by encode_bitmask."""
    if not isinstance(payload, dict):
        return None
    if payload.get("encoding") != "bitmask":
        return None
    try:
        height = int(payload["height"])
        width = int(payload["width"])
        num_bits = int(payload.get("num_bits", height * width))
        data = base64.b64decode(payload["data"])
    except (KeyError, ValueError, TypeError, base64.binascii.Error):
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(array)
    total = height * width
    if bits.size < total:
        bits = np.pad(bits, (0, total - bits.size), mode="constant")
    mask = bits[:total].reshape((height, width))
    return mask.astype(bool)


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """Compute an axis-aligned bounding box from a binary mask."""
    if mask.ndim != 2:
        raise ValueError("bbox_from_mask expects a 2D mask.")
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0).tolist()
    y_max, x_max = coords.max(axis=0).tolist()
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def encode_depth_map(depth_map: np.ndarray) -> Dict[str, object]:
    """Encode a depth map (float32/float64) as base64 float16 payload."""
    if depth_map.ndim != 2:
        raise ValueError("encode_depth_map expects a 2D depth map.")
    height, width = depth_map.shape
    depth = np.asarray(depth_map, dtype=np.float16)
    encoded = base64.b64encode(depth.tobytes()).decode("ascii")
    return {
        "encoding": "float16",
        "height": int(height),
        "width": int(width),
        "data": encoded,
    }


def decode_depth_map(
    payload: Optional[Dict[str, object]]
) -> Optional[np.ndarray]:
    """Decode a depth map produced by encode_depth_map."""
    if not isinstance(payload, dict):
        return None
    if payload.get("encoding") != "float16":
        return None
    try:
        height = int(payload["height"])
        width = int(payload["width"])
        data = base64.b64decode(payload["data"])
    except (KeyError, ValueError, TypeError, base64.binascii.Error):
        return None
    arr = np.frombuffer(data, dtype=np.float16)
    if arr.size < height * width:
        return None
    depth = arr.reshape((height, width)).astype(np.float32)
    return depth


def encode_depth_map_uint16(depth_map: np.ndarray) -> Dict[str, object]:
    """Encode a depth map (float32/float64) as uint16 png"""
    if depth_map.ndim != 2:
        raise ValueError("encode_depth_map_uint16 expects a 2D depth map.")
    height, width = depth_map.shape
    
    # Ensure we have numeric data; replace NaNs/Infs with zero and clip into range.
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    depth = (depth * 1000).astype(np.uint16)        # convert m to mm
    depth = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    
    # PNG stores uint16 in big-endian order; Pillow expects little-endian, so enforce it.
    depth = depth.view(dtype="<u2")
    
    buffer = io.BytesIO()
    Image.fromarray(depth, mode="I;16").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "encoding": "uint16_png",
        "height": int(height),
        "width": int(width),
        "data": encoded,
    }


def decode_depth_map_uint16(
    payload: Optional[Dict[str, object]]
) -> Optional[np.ndarray]:
    """Decode a depth map produced by encode_depth_map_uint16."""
    if not isinstance(payload, dict):
        return None
    if payload.get("encoding") != "uint16_png":
        return None
    try:
        height = int(payload["height"])
        width = int(payload["width"])
        data = base64.b64decode(payload["data"])
    except (KeyError, ValueError, TypeError, base64.binascii.Error):
        return None

    buffer = io.BytesIO(data)
    try:
        with Image.open(buffer) as image:
            if image.mode not in ("I;16", "I;16L", "I;16B"):
                image = image.convert("I;16")
            depth_uint16 = np.array(image, dtype=np.uint16, copy=False)
    except (OSError, ValueError):
        return None

    if depth_uint16.shape != (height, width):
        return None

    depth_uint16 = depth_uint16.astype(np.dtype("<u2"), copy=False)
    depth = depth_uint16.astype(np.float32) / 1000.0
    return depth


def segmentation_stub(
    stats: Dict[str, float], segments: int = 3
) -> List[Dict[str, object]]:
    """Generate deterministic segmentation metadata with encoded masks."""
    height = max(1, int(round(stats["height"])))
    width = max(1, int(round(stats["width"])))
    total_area = float(height * width)
    segments = max(1, segments)

    results: List[Dict[str, object]] = []
    stripe_width = max(1, width // (segments + 1))
    for idx in range(segments):
        mask = np.zeros((height, width), dtype=np.uint8)
        start_x = idx * stripe_width
        end_x = min(width, start_x + stripe_width * 2)
        mask[:, start_x:end_x] = 1
        coverage = float(mask.mean())
        area = float(mask.sum())
        payload = encode_bitmask(mask)
        bbox = bbox_from_mask(mask)
        results.append(
            {
                "id": idx,
                "label": f"segment_{idx}",
                "score": float(
                    round(0.55 + 0.35 * (idx + 1) / (segments + 1), 4)
                ),
                "coverage_ratio": float(round(coverage, 4)),
                "area": float(round(area, 2)),
                "bbox": bbox,
                "mask": payload,
            }
        )
    return results


def depth_stub(stats: Dict[str, float]) -> Dict[str, object]:
    """Produce deterministic depth statistics with an encoded pseudo depth map."""
    height = max(1, int(round(stats["height"])))
    width = max(1, int(round(stats["width"])))
    x = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num=height, dtype=np.float32).reshape(-1, 1)
    depth_map = (x * 0.6 + y * 0.4) * (stats["mean"] / 255.0 + 0.1)
    depth_map = np.clip(depth_map, 0.0, 1.5)
    depth_mean = float(depth_map.mean())
    depth_max = float(depth_map.max())
    depth_min = float(depth_map.min())
    payload = encode_depth_map(depth_map)
    return {
        "mean_depth": float(round(depth_mean, 4)),
        "min_depth": float(round(depth_min, 4)),
        "max_depth": float(round(depth_max, 4)),
        "map": payload,
    }


def iterate_images_from_input(
    input_path: Union[str, Path],
    *,
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[LanceBatch], List[np.ndarray]]:
    """Return Lance batch when applicable and a list of images to process."""
    if isinstance(input_path, str) and (
        input_path.startswith("http://") or input_path.startswith("https://")
    ):
        return None, [load_image(input_path)]

    path = Path(input_path)
    if path.suffix == ".lance":
        parameters = dict(lance_kwargs or {})
        image_col = parameters.pop("image_col", "camera_left")
        batch = load_lance_batch(
            path, image_col=image_col, reader_kwargs=parameters
        )
        return batch, list(batch.images)
    if path.is_dir():
        reader = Input(str(path), name="imgdir").get_reader()
        images = [next(reader) for _ in range(len(reader))]
        return None, images
    return None, [load_image(path)]
