"""OTN inference node for GroundingDINO object detection.

deps:
    transformers
    pylance
    flash-attn
    qwen-vl-utils
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Optional, Union

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.lance_utils import (
    compute_image_stats,
    iterate_images_from_input,
    scaled_bbox,
    write_json,
    write_lance_with_column,
)
from pdebug.utils.env import TORCH_INSTALLED

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


def _dummy_detections(
    stats: Dict[str, float], index: int
) -> Dict[str, object]:
    primary = scaled_bbox(stats, 0.6)
    secondary = scaled_bbox(stats, 0.3)
    detections = [
        {
            "label": f"object_{index}_primary",
            "score": 0.75,
            "bbox": primary,
        },
        {
            "label": f"object_{index}_secondary",
            "score": 0.42,
            "bbox": secondary,
        },
    ]
    return {
        "detections": detections,
        "model_name": "GroundingDINO",
        "processing_time_ms": 0.0,
    }


@lru_cache(maxsize=1)
def _load_groundingdino_model():
    """Load GroundingDINO model and processor from transformers."""
    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for GroundingDINO inference.")

    try:  # pragma: no cover - heavy dependency path
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install 'transformers' package to run GroundingDINO inference."
        ) from exc

    model_id = "IDEA-Research/grounding-dino-base"
    has_cuda = torch.cuda.is_available()
    default_device = "cuda" if has_cuda else "cpu"
    device_map = "auto" if torch.cuda.device_count() > 1 else default_device

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_id)
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
        "prompt": "all objects.",
        "box_threshold": 0.3,
        "text_threshold": 0.3,
    }
    return model, processor, settings


def _grounding_infer(
    image, *, index: int, unittest: bool
) -> Dict[str, object]:
    stats = compute_image_stats(image)
    if unittest:
        return _dummy_detections(stats, index)

    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for GroundingDINO inference.")

    import torch

    model, processor, settings = _load_groundingdino_model()
    device = torch.device(settings["device"])

    if Image is not None and not isinstance(image, Image.Image):
        image_input = Image.fromarray(image)
    else:
        image_input = image

    inputs = processor(
        images=image_input,
        text=settings["prompt"],
        return_tensors="pt",
        padding="longest",
    )

    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:  # pragma: no cover - defensive
            model_dtype = torch.float32

    tensor_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved = value.to(device)
            tensor_inputs[key] = (
                moved.to(model_dtype) if moved.is_floating_point() else moved
            )
        else:
            tensor_inputs[key] = value

    with torch.no_grad():
        outputs = model(**tensor_inputs)

    height, width = image_input.height, image_input.width
    target_sizes = torch.tensor([[height, width]], device=device)
    processed = processor.post_process_grounded_object_detection(
        outputs,
        tensor_inputs.get("input_ids"),
        threshold=settings["box_threshold"],
        text_threshold=settings["text_threshold"],
        target_sizes=target_sizes,
    )[0]

    boxes_tensor = processed["boxes"].to("cpu")
    scores_tensor = processed["scores"].to("cpu")
    labels_data = processed.get("text_labels")

    if labels_data is None:
        label_ids = processed.get("labels")
        if label_ids is not None:
            label_ids = label_ids.to("cpu").tolist()
            labels_data = [
                model.config.id2label.get(idx, f"object_{index}_{i}")
                for i, idx in enumerate(label_ids)
            ]
        else:
            labels_data = [
                f"object_{index}_{i}" for i in range(len(scores_tensor))
            ]

    detections = []
    for i, (box, score, label) in enumerate(
        zip(boxes_tensor.tolist(), scores_tensor.tolist(), labels_data)
    ):
        detections.append(
            {
                "label": str(label),
                "score": float(round(score, 4)),
                "bbox": [float(round(v, 4)) for v in box],
            }
        )
    return {
        "detections": detections,
        "model_name": settings["model_id"],
        "processing_time_ms": 0.0,
    }


@otn_manager.NODE.register(name="groundingdino")
def groundingdino_node(
    input_path: str,
    output: Optional[str] = None,
    *,
    output_key: str = "cortexia_detection",
    unittest: bool = False,
    lance_image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> Union[Dict[str, object], str]:
    """Run GroundingDINO over an image path or Lance dataset."""
    lance_kwargs = lance_kwargs or {}
    batch, images = iterate_images_from_input(
        input_path, lance_kwargs={"image_col": lance_image_col, **lance_kwargs}
    )

    if batch is not None:
        if output is None:
            raise ValueError(
                "`output` must be provided for Lance dataset inference."
            )
        results: List[Dict[str, object]] = []
        for idx, image in enumerate(images):
            results.append(
                _grounding_infer(image, index=idx, unittest=unittest)
            )
        written = write_lance_with_column(
            batch.table, output_key, results, output
        )
        return str(written)

    if not images:
        raise RuntimeError(f"No images found at {input_path}")
    result = _grounding_infer(images[0], index=0, unittest=unittest)
    if output:
        write_json(output, result)
        return output
    return result


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(groundingdino_node)
