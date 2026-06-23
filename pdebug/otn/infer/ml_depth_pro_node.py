"""OTN inference node for Apple's ML-Depth-Pro model."""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, Optional, Union

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.io_adapters import run_image_input
from pdebug.otn.inference import ImageInferenceRunner
from pdebug.piata import compute_image_stats, depth_stub, encode_depth_map
from pdebug.utils.env import TORCH_INSTALLED

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


def _dummy_depth(stats: Dict[str, float]) -> Dict[str, object]:
    depth = depth_stub(stats)
    return {
        "model_name": "ml-depth-pro",
        "processing_time_ms": 0.0,
        **depth,
    }


@lru_cache(maxsize=1)
def _load_depth_pro_model():
    """Load DepthPro model and image processor from transformers."""
    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for ML-Depth-Pro inference.")

    try:  # pragma: no cover - heavy dependency path
        import torch
        from transformers import (
            DepthProForDepthEstimation,
            DepthProImageProcessorFast,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install 'transformers' package to run DepthPro inference."
        ) from exc

    model_id = "apple/DepthPro-hf"
    has_cuda = torch.cuda.is_available()
    default_device = "cuda" if has_cuda else "cpu"
    device_map = "auto" if torch.cuda.device_count() > 1 else default_device

    processor = DepthProImageProcessorFast.from_pretrained(model_id)
    model = DepthProForDepthEstimation.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=torch.float16 if has_cuda else torch.float32,
    )
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
    }
    return model, processor, settings


_original_depth_cache_clear = _load_depth_pro_model.cache_clear


def _depth_cache_clear() -> None:
    if _load_depth_pro_model.cache_info().currsize:
        try:
            model, _, _ = _load_depth_pro_model()
            if TORCH_INSTALLED:
                import gc

                import torch

                if torch.cuda.is_available():
                    try:
                        model.to("cpu")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                        if callable(ipc_collect):
                            ipc_collect()
                    except Exception:
                        pass
                del model
                gc.collect()
        except Exception:
            pass
    _original_depth_cache_clear()
    if TORCH_INSTALLED:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except Exception:
            pass


_load_depth_pro_model.cache_clear = _depth_cache_clear  # type: ignore[assignment]


def _depth_infer(
    image, *, index: int = 0, unittest: bool
) -> Dict[str, object]:
    stats = compute_image_stats(image)
    if unittest:
        return _dummy_depth(stats)

    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for ML-Depth-Pro inference.")

    import torch

    model, processor, settings = _load_depth_pro_model()
    device = torch.device(settings["device"])

    if Image is not None and not isinstance(image, Image.Image):
        image_input = Image.fromarray(image)
    else:
        image_input = image

    inputs = processor(images=image_input, return_tensors="pt")
    tensor_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and key == "pixel_values":
            tensor_inputs[key] = value.to(device)
        else:
            tensor_inputs[key] = value

    with torch.no_grad():
        outputs = model(**tensor_inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image_input.height, image_input.width)],
    )[0]

    depth_map = post_processed["predicted_depth"].detach().cpu().numpy()
    depth = {
        "mean_depth": float(depth_map.mean()),
        "min_depth": float(depth_map.min()),
        "max_depth": float(depth_map.max()),
        "model_name": settings["model_id"],
        "processing_time_ms": 0.0,
        "map": encode_depth_map(depth_map),
    }
    focal_length = post_processed.get("focal_length")
    if focal_length is not None:
        depth["focal_length_px"] = float(focal_length.detach().cpu().item())
    return depth


@otn_manager.NODE.register(name="ml_depth_pro")
def ml_depth_pro_node(
    input_path: str,
    output: Optional[str] = None,
    *,
    output_key: str = "cortexia_depth",
    unittest: bool = False,
    lance_image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> Union[Dict[str, object], str]:
    """Run ML-Depth-Pro over an image path or Lance dataset."""
    return run_image_input(
        ImageInferenceRunner(_depth_infer),
        input_path,
        output,
        unittest=unittest,
        output_key=output_key,
        lance_image_col=lance_image_col,
        lance_kwargs=lance_kwargs,
        missing_output_message=(
            "`output` must be specified for Lance dataset outputs."
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(ml_depth_pro_node)
