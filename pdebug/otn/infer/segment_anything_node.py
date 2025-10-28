"""OTN inference node for Segment Anything."""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Optional, Union

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.lance_utils import (
    bbox_from_mask,
    compute_image_stats,
    encode_bitmask,
    iterate_images_from_input,
    segmentation_stub,
    write_json,
    write_lance_with_column,
)
from pdebug.utils.env import TORCH_INSTALLED

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


def _dummy_segments(stats: Dict[str, float], index: int) -> Dict[str, object]:
    segments = segmentation_stub(stats, segments=3)
    for seg in segments:
        seg["label"] = f"{seg['label']}_{index}"
    return {
        "segmentations": segments,
        "model_name": "SegmentAnything",
        "processing_time_ms": 0.0,
    }


@lru_cache(maxsize=1)
def _load_segment_anything_model():
    """Load SAM model and processor via transformers."""
    if not TORCH_INSTALLED:
        raise ImportError(
            "PyTorch is required for Segment Anything inference."
        )

    try:  # pragma: no cover - heavy dependency path
        import torch
        from transformers import SamModel, SamProcessor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install 'transformers' package to run Segment Anything inference."
        ) from exc

    model_id = "facebook/sam-vit-base"
    has_cuda = torch.cuda.is_available()
    default_device = "cuda" if has_cuda else "cpu"
    device_map = "auto" if torch.cuda.device_count() > 1 else default_device

    model = SamModel.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float32,
    )
    processor = SamProcessor.from_pretrained(model_id)
    model.eval()

    if hasattr(model, "hf_device_map") and isinstance(
        model.hf_device_map, dict
    ):
        first_device = next(iter(model.hf_device_map.values()))
        device = str(first_device)
    else:
        device = default_device

    settings = {
        "model_id": model_id,
        "device": device,
        "mask_threshold": 0.0,
    }
    return model, processor, settings


def _segment_anything_infer(
    image, *, index: int, unittest: bool
) -> Dict[str, object]:
    stats = compute_image_stats(image)
    if unittest:
        return _dummy_segments(stats, index)

    if not TORCH_INSTALLED:
        raise ImportError(
            "PyTorch is required for Segment Anything inference."
        )

    import torch

    model, processor, settings = _load_segment_anything_model()
    device = torch.device(settings["device"])

    if Image is not None and not isinstance(image, Image.Image):
        image_input = Image.fromarray(image)
    else:
        image_input = image

    width, height = image_input.size
    points = [
        [
            [width * 0.5, height * 0.5],
            [width * 0.25, height * 0.25],
            [width * 0.75, height * 0.75],
        ]
    ]
    labels = [[1, 1, 1]]

    inputs = processor(
        images=image_input,
        input_points=points,
        input_labels=labels,
        return_tensors="pt",
    )

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:  # pragma: no cover - defensive
            model_dtype = torch.float32

    tensor_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and key not in {
            "original_sizes",
            "reshaped_input_sizes",
        }:
            moved = value.to(device)
            tensor_inputs[key] = (
                moved.to(model_dtype) if moved.is_floating_point() else moved
            )
        else:
            tensor_inputs[key] = value

    with torch.no_grad():
        outputs = model(**tensor_inputs, multimask_output=True)

    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=tensor_inputs["original_sizes"],
        reshaped_input_sizes=tensor_inputs["reshaped_input_sizes"],
    )[0][
        0
    ]  # first image, first prompt set

    scores = outputs.iou_scores[0, 0].detach().cpu().tolist()

    segmentations = []
    for idx_mask, (mask_tensor, score) in enumerate(zip(masks, scores)):
        binary_mask = (mask_tensor > settings["mask_threshold"]).to(
            dtype=torch.int32
        )
        mask_np = binary_mask.detach().cpu().numpy().astype(bool)
        area = float(mask_np.sum())
        coverage = area / float(mask_np.size)
        bbox = bbox_from_mask(mask_np)
        segmentations.append(
            {
                "label": f"segment_{idx_mask}",
                "score": float(round(score, 4)),
                "area": float(area),
                "bbox": bbox,
                "mask": encode_bitmask(mask_np),
                "coverage_ratio": float(round(coverage, 4)),
            }
        )

    return {
        "segmentations": segmentations,
        "model_name": settings["model_id"],
        "processing_time_ms": 0.0,
    }


@otn_manager.NODE.register(name="segment_anything")
def segment_anything_node(
    input_path: str,
    output: Optional[str] = None,
    *,
    output_key: str = "cortexia_segmentation",
    unittest: bool = False,
    lance_image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> Union[Dict[str, object], str]:
    """Run Segment Anything over an image path or Lance dataset."""
    lance_kwargs = lance_kwargs or {}
    batch, images = iterate_images_from_input(
        input_path, lance_kwargs={"image_col": lance_image_col, **lance_kwargs}
    )

    if batch is not None:
        if output is None:
            raise ValueError(
                "`output` must be provided when saving Lance results."
            )
        results: List[Dict[str, object]] = []
        for idx, image in enumerate(images):
            results.append(
                _segment_anything_infer(image, index=idx, unittest=unittest)
            )
        written = write_lance_with_column(
            batch.table, output_key, results, output
        )
        return str(written)

    if not images:
        raise RuntimeError(f"No images found at {input_path}")
    result = _segment_anything_infer(images[0], index=0, unittest=unittest)
    if output:
        write_json(output, result)
        return output
    return result


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(segment_anything_node)
