import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.lance_utils import (
    compute_image_stats,
    iterate_images_from_input,
    write_lance_with_column,
)
from pdebug.piata import Input
from pdebug.utils.env import TORCH_INSTALLED

import cv2
import numpy as np
import typer

if TORCH_INSTALLED:
    import torch

DEFAULT_IMAGE_PATH = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)
DEFAULT_TEXT = "Describe the image."


def _dummy_tags(stats: Dict[str, float], index: int) -> Dict[str, object]:
    base = int(stats["mean"]) % 97
    width_tag = f"width_{int(stats['width'])}"
    tags = [
        f"tag_{index}_{base}",
        width_tag,
        f"std_{int(stats['std']) % 53}",
    ]
    raw_response = ", ".join(tags)
    return {
        "tags": tags,
        "raw_response": raw_response,
        "confidence_scores": None,
        "model_name": "Qwen2.5-VL",
        "processing_time_ms": 0.0,
    }


@lru_cache(maxsize=1)
def _load_qwen_model():
    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for Qwen2.5-VL inference.")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def _normalize_tags(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("\n", ", ").replace(" and ", ", ")
    parts = [
        part.strip() for part in re.split(r"[,;]\s*", cleaned) if part.strip()
    ]
    if not parts:
        stripped = text.strip()
        return [stripped] if stripped else []
    return parts


def _save_temp_image(image, target: Path) -> None:
    if cv2 is None:
        from PIL import Image

        Image.fromarray(image).save(target)
        return
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(target), bgr)


def _real_qwen_dataset_infer(
    images: Sequence[np.ndarray],
    *,
    prompt: str,
) -> List[Dict[str, object]]:
    if not TORCH_INSTALLED:
        raise ImportError("PyTorch is required for Qwen2.5-VL inference.")
    from qwen_vl_utils import process_vision_info

    model, processor = _load_qwen_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, image in enumerate(images):
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "Expected numpy.ndarray images from Lance reader."
                )
            temp_path = tmpdir_path / f"frame_{idx:06d}.png"
            _save_temp_image(image, temp_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(temp_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            conversation = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[conversation],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            start = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw_response = outputs[0] if outputs else ""
            tags = _normalize_tags(raw_response)

            results.append(
                {
                    "tags": tags,
                    "raw_response": raw_response,
                    "confidence_scores": None,
                    "model_name": "Qwen2.5-VL",
                    "processing_time_ms": float(round(elapsed_ms, 3)),
                }
            )

    return results


def lance_dataset_infer(
    input_path: str,
    output: Optional[str],
    *,
    unittest: bool,
    text: str = DEFAULT_TEXT,
    output_key: str = "cortexia_tags",
    lance_kwargs: Optional[Dict[str, object]] = None,
) -> str:
    if output is None:
        raise ValueError(
            "`output` must be provided for Lance dataset inference."
        )
    lance_parameters = {"image_col": "camera_left", **(lance_kwargs or {})}
    batch, images = iterate_images_from_input(
        input_path, lance_kwargs=lance_parameters
    )
    if batch is None:
        raise RuntimeError(f"Failed to load Lance dataset from {input_path}")

    if unittest:
        results: List[Dict[str, object]] = []
        for idx, image in enumerate(images):
            stats = compute_image_stats(image)
            results.append(_dummy_tags(stats, idx))
    else:
        results = _real_qwen_dataset_infer(images, prompt=text)

    written = write_lance_with_column(batch.table, output_key, results, output)
    return str(written)


@otn_manager.NODE.register(name="qwen2_5_vl")
def main(
    input_path: str = DEFAULT_IMAGE_PATH,
    text: str = DEFAULT_TEXT,
    output: str = None,
    unittest: bool = False,
    cache: bool = True,
):
    """Infer using qwen2.5-vl"""
    if isinstance(input_path, str) and input_path.endswith(".lance"):
        return lance_dataset_infer(
            input_path,
            output,
            unittest=unittest,
            text=text,
        )

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not unittest:
        input_path = Path(input_path)
        output = Path(output)
        if cache and output.exists():
            print(f"{output} exists, skip")
            return
        output.parent.mkdir(parents=True, exist_ok=True)

    model, processor = _load_qwen_model()

    if unittest:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_path,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)

        # Inference: Generation of the output
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(output_text)
        return

    if input_path.is_dir():
        reader = Input(str(input_path), name="imgdir").get_reader()
        if len(reader) == 0:
            raise RuntimeError("No RGB files found")
        typer.echo(
            typer.style(
                f"Found {len(reader)} RGB files", fg=typer.colors.GREEN
            )
        )
    elif input_path.is_file():
        reader = Input(str(input_path), name="video").get_reader()
        typer.echo(
            typer.style(
                f"Found {len(reader)} frames in video", fg=typer.colors.GREEN
            )
        )
    else:
        raise RuntimeError("No RGB files or video found")

    image = next(reader)
    temp_dir = tempfile.TemporaryDirectory()
    image_file = os.path.join(temp_dir.name, "input.png")
    cv2.imwrite(image_file, image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": text},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    # Inference: Generation of the output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    typer.echo(typer.style(f">> {text}", fg=typer.colors.YELLOW))
    typer.echo(typer.style(f">> {output_text[0]}", fg=typer.colors.GREEN))

    with open(output, "w") as fid:
        fid.write(output_text[0])
    temp_dir.cleanup()
    typer.echo(f"Output saved to {output}")


if __name__ == "__main__":
    typer.run(main)
