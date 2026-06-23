"""Generic image helpers for data IO and lightweight inference outputs."""
from __future__ import annotations
import base64
import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "bbox_from_mask",
    "compute_image_stats",
    "decode_bitmask",
    "decode_depth_map",
    "decode_depth_map_uint16",
    "decode_png_u8",
    "depth_stub",
    "deterministic_caption",
    "encode_bitmask",
    "encode_depth_map",
    "encode_depth_map_uint16",
    "encode_png_u8",
    "load_image",
    "scaled_bbox",
    "segmentation_stub",
]


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
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
    """Encode a 2D uint8 mask as PNG bytes."""
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
    """Decode PNG bytes into a 2D uint8 mask."""
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
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
    """Encode a depth map as base64 float16 payload."""
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
    return arr.reshape((height, width)).astype(np.float32)


def encode_depth_map_uint16(depth_map: np.ndarray) -> Dict[str, object]:
    """Encode a depth map as uint16 PNG payload."""
    if depth_map.ndim != 2:
        raise ValueError("encode_depth_map_uint16 expects a 2D depth map.")
    height, width = depth_map.shape
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    depth = (depth * 1000).astype(np.uint16)
    depth = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(
        np.uint16, copy=False
    )
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
    return depth_uint16.astype(np.float32) / 1000.0


def segmentation_stub(
    stats: Dict[str, float], segments: int = 3
) -> List[Dict[str, object]]:
    """Generate deterministic segmentation metadata with encoded masks."""
    height = max(1, int(round(stats["height"])))
    width = max(1, int(round(stats["width"])))
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
        results.append(
            {
                "id": idx,
                "label": f"segment_{idx}",
                "score": float(
                    round(0.55 + 0.35 * (idx + 1) / (segments + 1), 4)
                ),
                "coverage_ratio": float(round(coverage, 4)),
                "area": float(round(area, 2)),
                "bbox": bbox_from_mask(mask),
                "mask": encode_bitmask(mask),
            }
        )
    return results


def depth_stub(stats: Dict[str, float]) -> Dict[str, object]:
    """Produce deterministic depth statistics with encoded pseudo depth map."""
    height = max(1, int(round(stats["height"])))
    width = max(1, int(round(stats["width"])))
    x = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num=height, dtype=np.float32).reshape(-1, 1)
    depth_map = (x * 0.6 + y * 0.4) * (stats["mean"] / 255.0 + 0.1)
    depth_map = np.clip(depth_map, 0.0, 1.5)
    return {
        "mean_depth": float(round(float(depth_map.mean()), 4)),
        "min_depth": float(round(float(depth_map.min()), 4)),
        "max_depth": float(round(float(depth_map.max()), 4)),
        "map": encode_depth_map(depth_map),
    }
