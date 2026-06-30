"""OTN inference node for the Moondream multimodal model."""
from __future__ import annotations
import time
from functools import lru_cache
from typing import Dict, Optional, Union

from pdebug.otn.infer.base import ImageInferenceRunner
from pdebug.otn.infer.io_adapters import run_image_input
from pdebug.piata import compute_image_stats, deterministic_caption
from pdebug.utils.env import TORCH_INSTALLED, TRANSFORMERS_INSTALLED

from packaging.version import Version

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


def _dummy_caption(stats: Dict[str, float], index: int) -> Dict[str, object]:
    caption = deterministic_caption(stats, prefix=f"moondream-{index}")
    return {
        "caption": caption,
        "confidence": None,
        "model_name": "moondream",
        "caption_length": str(len(caption)),
        "processing_time_ms": 0.0,
    }


def _moondream_infer(
    image, *, index: int, unittest: bool
) -> Dict[str, object]:
    stats = compute_image_stats(image)
    if unittest:
        return _dummy_caption(stats, index)

    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for Moondream inference.")

    model, settings = _load_moondream_model()
    device = settings["device"]
    length = settings["caption_length"]

    if Image is not None and not isinstance(image, Image.Image):
        image_input = Image.fromarray(image)
    else:
        image_input = image

    start = time.perf_counter()
    result = model.caption(image_input, length=length, stream=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    caption = (
        result.get("caption") if isinstance(result, dict) else str(result)
    )

    return {
        "caption": caption,
        "confidence": None,
        "model_name": settings["model_id"],
        "caption_length": length,
        "processing_time_ms": float(round(elapsed_ms, 3)),
    }


@lru_cache(maxsize=1)
def _load_moondream_model():
    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for Moondream inference.")
    if not TRANSFORMERS_INSTALLED:
        raise ImportError("transformers is required for Moondream inference.")

    import torch
    from transformers import AutoModelForCausalLM  # type: ignore

    if Version(torch.__version__) < Version("2.5.0"):
        raise RuntimeError(
            f"PyTorch version ({torch.__version__}) should >= 2.5.0"
        )

    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"
    has_cuda = torch.cuda.is_available()
    default_device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    device_map = "auto" if torch.cuda.device_count() > 1 else default_device

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    # Determine device where primary weights live
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
        "caption_length": "long",
    }
    return model, settings


_original_moondream_clear = _load_moondream_model.cache_clear


def _moondream_cache_clear() -> None:
    if _load_moondream_model.cache_info().currsize:
        try:
            model, _ = _load_moondream_model()
            if TORCH_INSTALLED:
                import gc

                import torch

                try:
                    if hasattr(model, "to"):
                        model.to("cpu")
                except Exception:
                    pass
                if torch.cuda.is_available():
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
    _original_moondream_clear()
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


_load_moondream_model.cache_clear = _moondream_cache_clear  # type: ignore[assignment]


def moondream_node(
    input_path: str,
    output: Optional[str] = None,
    *,
    output_key: str = "cortexia_caption",
    unittest: bool = False,
    lance_image_col: str = "camera_left",
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> Union[Dict[str, object], str]:
    """Run Moondream inference over an image or Lance dataset.

    torch >= 2,5.0 is required.
    """
    return run_image_input(
        ImageInferenceRunner(_moondream_infer),
        input_path,
        output,
        unittest=unittest,
        output_key=output_key,
        lance_image_col=lance_image_col,
        lance_kwargs=lance_kwargs,
    )


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(moondream_node)
