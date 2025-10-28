from typing import Any, Dict, List

import numpy as np

__all__ = [
    "get_value_of_one_item_dict",
    "update_dict",
    "segmentation_to_mask",
    "as_list_or_tensor",
]


def get_value_of_one_item_dict(item):
    assert isinstance(item, dict)
    assert len(item) == 1
    value = [item[k] for k in item][0]
    return value


def update_dict(d1, d2):
    """Update d1 using d2, recursively."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            update_dict(d1[k], v)
        else:
            d1[k] = v


def segmentation_to_mask(
    roi: Dict[str, Any], min_area: float = 100
) -> np.ndarray:
    """Convert roi segmentation to mask"""
    from pdebug.data_types import Segmentation

    return Segmentation.to_mask(
        roi["segmentation"],
        roi["category_id"],
        roi["image_height"],
        roi["image_width"],
        min_area=100,
    )
    return

    """
    from imantics import Polygons

    assert "image_height" in roi
    assert "image_width" in roi
    assert "segmentation" in roi
    assert "category_id" in roi

    h = roi["image_height"]
    w = roi["image_width"]
    mask_merged = np.zeros((h, w), dtype=np.int32)
    for points, cat_id in zip(roi["segmentation"], roi["category_id"]):
        if np.asarray(points).ndim == 1:
            points = [points]
        polygons = Polygons(points)
        mask = polygons.mask(w, h).array
        if mask.sum() < min_area:
            continue
        mask_merged[mask != 0] = cat_id - 1 # category in coco is one-added
    return mask_merged
    """


def as_list_or_tensor(x: Any) -> List:
    if isinstance(x, list):
        return x
    elif hasattr(x, "shape") and hasattr(x, "ndim"):
        return x
    else:
        return [x]
