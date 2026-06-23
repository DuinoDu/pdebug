"""Piata is a data package, providing all data processing utilities you need."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.0.2"
__author__ = "duinodu"
__licence__ = ""
__homepage__ = ""
__docs__ = ""
__long_docs__ = ""

from .image_io import (
    ImageBatch,
    load_lance_batch,
    read_image_batch,
    write_json_result,
    write_lance_column,
)
from .input import Input
from .output import Output

__all__ = [
    "ImageBatch",
    "Input",
    "Output",
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
    "load_lance_batch",
    "read_image_batch",
    "scaled_bbox",
    "segmentation_stub",
    "write_json_result",
    "write_lance_column",
]

_IMAGE_UTIL_NAMES = {
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
}

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    from . import handler as handler_module  # noqa: F401
    from .image_utils import (
        bbox_from_mask,
        compute_image_stats,
        decode_bitmask,
        decode_depth_map,
        decode_depth_map_uint16,
        decode_png_u8,
        depth_stub,
        deterministic_caption,
        encode_bitmask,
        encode_depth_map,
        encode_depth_map_uint16,
        encode_png_u8,
        load_image,
        scaled_bbox,
        segmentation_stub,
    )


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules on first access."""
    if name == "handler":
        module = import_module("pdebug.piata.handler")
        globals()[name] = module
        return module
    if name in _IMAGE_UTIL_NAMES:
        module = import_module("pdebug.piata.image_utils")
        for export_name in _IMAGE_UTIL_NAMES:
            globals()[export_name] = getattr(module, export_name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
