"""OTN inference node for the Moondream multimodal model."""
from __future__ import annotations
import time
from functools import lru_cache
from typing import Dict, List, Optional, Union

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.lance_utils import (
    compute_image_stats,
    deterministic_caption,
    iterate_images_from_input,
    write_json,
    write_lance_with_column,
)
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

    if Version(torch.__version__) < Version("2.7.0"):
        raise RuntimeError(
            f"PyTorch version ({torch.__version__}) should >= 2.7.0"
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
        torch_dtype=dtype,
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


@otn_manager.NODE.register(name="moondream")
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
    lance_kwargs = lance_kwargs or {}
    batch, images = iterate_images_from_input(
        input_path, lance_kwargs={"image_col": lance_image_col, **lance_kwargs}
    )

    if batch is not None:
        if output is None:
            raise ValueError(
                "`output` is required when writing Lance results."
            )
        results: List[Dict[str, object]] = []
        for idx, image in enumerate(images):
            results.append(
                _moondream_infer(image, index=idx, unittest=unittest)
            )
        written = write_lance_with_column(
            batch.table, output_key, results, output
        )
        return str(written)

    if not images:
        raise RuntimeError(f"No images found at {input_path}")
    result = _moondream_infer(images[0], index=0, unittest=unittest)
    if output:
        write_json(output, result)
        return output
    return result


if __name__ == "__main__":  # pragma: no cover
    import typer

    typer.run(moondream_node)
