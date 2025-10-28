import json
import os
import shutil
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.fileio import save_json
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer
from PIL import Image


def installed():
    try:
        import langsam_sam

        return True
    except ModuleNotFoundError as e:
        return False


@otn_manager.NODE.register(name="langsam_sam")
def langsam_sam(
    input_path: str,
    output: str = None,
    sam_type: str = "sam2.1_hiera_small",
    ckpt_path: str = None,
    device: str = "cuda",
    use_auto_mask: bool = False,
    topk: int = -1,
):
    """SAM segmentation infer node.

    Args:
        input_path: Path to RGB images folder
        output: Output directory for segmentation results
        sam_type: SAM model type (sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large)
        ckpt_path: Optional custom checkpoint path
        device: Device to run inference on (cuda/cpu)
        use_auto_mask: Whether to use automatic mask generation (no prompts)
    """
    if not output:
        output = "tmp_sam_output"
    output = Path(output).resolve()
    output.mkdir(exist_ok=True, parents=True)
    input_path = Path(input_path)

    try:
        from lang_sam.models.sam import SAM
    except ImportError as e:
        raise ImportError(
            f"Failed to import lang_sam {e}, please install first."
        )

    # Initialize SAM model
    sam = SAM()
    typer.echo(f"Building SAM model: {sam_type}")
    sam.build_model(sam_type, ckpt_path, device=device)

    # Get input
    if input_path.is_dir():
        reader = Input(
            str(input_path), name="imgdir", to_rgb=True, topk=topk
        ).get_reader()
        if len(reader) == 0:
            raise RuntimeError("No RGB files found")
        typer.echo(
            typer.style(
                f"Found {len(reader)} RGB files", fg=typer.colors.GREEN
            )
        )
    elif input_path.is_file():
        reader = Input(
            str(input_path), name="video", to_rgb=True, topk=topk
        ).get_reader()
        typer.echo(
            typer.style(
                f"Found {len(reader)} frames in video", fg=typer.colors.GREEN
            )
        )
    else:
        raise RuntimeError("No RGB files or video found")

    # Process each image
    for idx, rgb in enumerate(reader):
        typer.echo(f"Processing {idx+1}/{len(reader)}")
        image_np = rgb
        image = Image.fromarray(rgb)

        if use_auto_mask:
            # Automatic mask generation
            masks = sam.generate(image_np)

            # Save masks
            mask_dir = output / "masks"
            mask_dir.mkdir(exist_ok=True)

            results = []
            for mask_idx, mask_info in enumerate(masks):
                mask = mask_info["segmentation"]
                mask_img = (mask * 255).astype(np.uint8)

                mask_file = (
                    mask_dir / f"{Path(rgb_file).stem}_mask_{mask_idx:03d}.png"
                )
                cv2.imwrite(str(mask_file), mask_img)

                # Save mask metadata
                mask_data = {
                    "file": str(mask_file.name),
                    "area": int(mask_info["area"]),
                    "bbox": mask_info["bbox"],  # [x, y, width, height]
                    "predicted_iou": float(mask_info["predicted_iou"]),
                    "stability_score": float(mask_info["stability_score"]),
                }
                results.append(mask_data)

            # Save metadata
            meta_file = output / "metadata" / f"{Path(rgb_file).stem}.json"
            meta_file.parent.mkdir(exist_ok=True)
            save_json(results, meta_file)

        else:
            raise NotImplementedError

    typer.echo(f"Results saved to {output}")


@otn_manager.NODE.register(name="langsam_predict")
def langsam_predict(
    input_path: str,
    output: str = None,
    texts: str = None,
    sam_type: str = "sam2.1_hiera_small",
    ckpt_path: str = None,
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    topk: int = -1,
    cache: bool = True,
):
    """Language-guided SAM segmentation infer node.

    Args:
        input_path: Path to RGB images folder
        texts: Text prompts (comma-separated for multiple prompts)
        output: Output directory for segmentation results
        sam_type: SAM model type
        ckpt_path: Optional custom checkpoint path
        device: Device to run inference on
        box_threshold: Threshold for box predictions from GDINO
        text_threshold: Threshold for text predictions from GDINO
    """
    if not output:
        output = "tmp_langs_output"
    output = Path(output).resolve()
    if (not cache) and output.exists():
        shutil.rmtree(output)
    output.mkdir(exist_ok=True, parents=True)
    input_path = Path(input_path)

    try:
        from lang_sam.lang_sam import LangSAM
    except ImportError as e:
        raise ImportError(
            f"Failed to import lang_sam from {repo_path}: {e}, install lang_sam first."
        )

    # Parse text prompts
    if not texts:
        texts = input_path.stem.split("__")[0]  # obj__idx, to remove idx
        typer.echo(
            typer.style(
                f"Inferred texts ({texts}) from input_path",
                fg=typer.colors.GREEN,
            )
        )
    elif os.path.exists(texts):
        lines = [l.strip() for l in open(texts, "r")]
        if not lines:
            raise RuntimeError(f"{texts} has no texts.")
        texts = lines[0]

    text_prompts = [t.strip() for t in texts.split(",")]
    typer.echo(f"Text prompts: {text_prompts}")

    # Get input
    if input_path.is_dir():
        reader = Input(
            str(input_path), name="imgdir", to_rgb=True, topk=topk
        ).get_reader()
        if len(reader) == 0:
            raise RuntimeError("No RGB files found")
        typer.echo(
            typer.style(
                f"Found {len(reader)} RGB files", fg=typer.colors.GREEN
            )
        )
    elif input_path.is_file():
        reader = Input(
            str(input_path), name="video", to_rgb=True, topk=topk
        ).get_reader()
        typer.echo(
            typer.style(
                f"Found {len(reader)} frames in video", fg=typer.colors.GREEN
            )
        )
    else:
        raise RuntimeError("No RGB files or video found")

    # Create output directories
    mask_dir = output / "masks"
    vis_dir = output / "vis"
    mask_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    if cache and len(os.listdir(mask_dir)) == len(reader):
        print(f"{output} exists, skip")
        return

    # Initialize LangSAM model
    lang_sam = LangSAM(sam_type=sam_type, ckpt_path=ckpt_path, device=device)

    # Process each image
    t = tqdm.tqdm(total=len(reader))
    for idx, rgb in enumerate(reader):
        t.update()
        image = Image.fromarray(rgb)

        # Get corresponding text prompt (cycle if fewer prompts than images)
        prompt = text_prompts[idx % len(text_prompts)]

        # Run inference
        results = lang_sam.predict(
            [image], [prompt], box_threshold, text_threshold
        )

        if results is None or len(results) == 0 or results[0]["masks"] is None:
            typer.echo(f"No objects detected for prompt '{prompt}'")
            continue

        result = results[0]

        # Save masks
        masks = result["masks"]
        boxes = result["boxes"]
        scores = result["scores"]

        rgb_file = reader.filename
        base_name = Path(rgb_file).stem

        # Save individual masks
        mask_files = []
        vis_mask_merge = np.zeros_like(masks[0]).astype(np.uint8)
        for mask_idx, mask in enumerate(masks):
            mask_img = (mask * 255).astype(np.uint8)
            mask_file = mask_dir / f"{base_name}_mask_{mask_idx:03d}.png"
            cv2.imwrite(str(mask_file), mask_img)
            mask_files.append(str(mask_file.name))
            vis_mask_merge[mask.astype(bool)] = mask_idx + 1

        # Save visualization
        vis_img = np.array(image)
        vis_img = draw.semseg(vis_mask_merge, vis_img)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                f"{prompt}:{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        vis_file = vis_dir / f"{base_name}_vis.jpg"
        cv2.imwrite(str(vis_file), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        # Save metadata
        metadata = {"image": str(rgb_file), "prompt": prompt, "detections": []}

        for i, (box, score) in enumerate(zip(boxes, scores)):
            detection = {
                "mask_file": mask_files[i],
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "score": float(score),
                "area": float(np.sum(masks[i])),
            }
            metadata["detections"].append(detection)

        meta_file = output / "metadata" / f"{base_name}.json"
        meta_file.parent.mkdir(exist_ok=True)
        save_json(metadata, meta_file)

    typer.echo(f"Results saved to {output}")


def expand_mask(mask, pixels):
    """对mask外扩pixels像素"""
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded


def crop_to_mask(image, mask, margin=0):
    """根据mask裁剪图片，margin为额外边距"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image, mask  # fallback
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, image.shape[1])
    y_max = min(y_max + margin, image.shape[0])
    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


@otn_manager.NODE.register(name="langsam_for_aigc")
def langsam_for_aigc(
    input_path=None,
    text="cube",
    output_path="center_object.png",
):
    """
    Generate object roi image from LangSAM.
    """
    try:
        from lang_sam import LangSAM
    except ImportError as e:
        raise ImportError(
            f"Failed to import lang_sam {e}, please install first."
        )
    input_path = Path(input_path)

    # 1. 加载图片
    if input_path.suffix == ".mp4":
        reader = Input(input_path, name="video", to_rgb=True).get_reader()
        raw_image = Image.fromarray(next(reader))
    elif Path(input_path).exists():
        raw_image = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Please provide image_path")

    if not text:
        text = input_path.stem.split("__")[0]  # obj__idx, to remove idx
        typer.echo(
            typer.style(
                f"Inferred text ({text}) from input_path",
                fg=typer.colors.GREEN,
            )
        )
    elif os.path.exists(text):
        lines = [l.strip() for l in open(text, "r")]
        if not lines:
            raise RuntimeError(f"{text} has no text.")
        text = lines[0]
    else:
        raise ValueError("Can not find text.")

    # 2. 加载LangSAM模型
    model = LangSAM()
    results = model.predict([raw_image], [text])

    # 3. 获取mask（取第一个结果的第一个mask）
    mask = results[0]["masks"][0]  # (H, W), bool
    mask = np.array(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask[0]
    mask = mask.astype(np.uint8)
    print("mask shape before expand:", mask.shape, mask.dtype)

    # 4. mask外扩20像素
    expanded_mask = expand_mask(mask, 0)

    # 5. 裁剪到mask区域
    image_np = np.array(raw_image)
    cropped_img, cropped_mask = crop_to_mask(
        image_np, expanded_mask, margin=20
    )

    # 6. resize到1024x1024（保持比例，padding）
    target_size = 1024
    # 计算缩放比例
    h, w = cropped_img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    # 缩放图片和mask
    cropped_img_pil = Image.fromarray(cropped_img).resize(
        (new_w, new_h), Image.LANCZOS
    )
    cropped_mask_pil = Image.fromarray(
        (cropped_mask * 255).astype(np.uint8)
    ).resize((new_w, new_h), Image.NEAREST)

    # 创建全透明底图和全0 mask
    padded_img = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    padded_mask = Image.new("L", (target_size, target_size), 0)
    # 计算padding起点
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    # 粘贴
    cropped_img_pil = cropped_img_pil.convert("RGBA")
    padded_img.paste(cropped_img_pil, (left, top))
    padded_mask.paste(cropped_mask_pil, (left, top))

    # 合成带透明背景的PNG
    cropped_mask_np = np.array(padded_mask)
    alpha = (cropped_mask_np > 127).astype(np.uint8) * 255
    rgba = np.array(padded_img)
    rgba[..., 3] = alpha
    out_img = Image.fromarray(rgba)
    out_img.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    typer.run(langsam_predict)
