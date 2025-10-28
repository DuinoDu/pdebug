import os
import sys
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.fileio import do_system
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer
from PIL import Image


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


@otn_manager.NODE.register(name="mask2rgba")
def main(
    input_path: str = None,
    mask_path: str = None,
    output: str = "tmp_mask2rgba",
    cache: bool = True,
    topk: int = -1,
    crop_margin: int = 20,
    keep_scale_and_fixed: bool = False,
    keep_scale_and_center: bool = True,
    target_size: int = 1024,
):
    """Convert image with mask to rgba files.

    Args:
        keep_scale_and_fixed: keep crop region with scale roi image, using a fixed bbox.
    """
    output = Path(output)
    output.mkdir(exist_ok=True)

    input_path = Path(input_path)
    mask_path = Path(mask_path)

    if input_path.is_dir():
        rgb_reader = Input(
            input_path, name="imgdir", topk=topk, to_rgb=True
        ).get_reader()
    else:
        rgb_reader = Input(
            input_path, name="video", topk=topk, to_rgb=True
        ).get_reader()
    mask_reader = Input(
        mask_path, name="imgdir", topk=topk, imread_raw=True
    ).get_reader()
    typer.echo(
        typer.style(f"loading {len(rgb_reader)} images", fg=typer.colors.GREEN)
    )

    if cache and len(os.listdir(output)) == len(rgb_reader):
        typer.echo(
            typer.style(f"{output} exists, skip", fg=typer.colors.YELLOW)
        )
        return

    if keep_scale_and_fixed:
        fixed_bbox = None
        for mask in mask_reader:
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            if fixed_bbox is None:
                fixed_bbox = [x_min, y_min, x_max, y_max]
            else:
                fixed_bbox[0] = min(x_min, fixed_bbox[0])
                fixed_bbox[1] = min(y_min, fixed_bbox[1])
                fixed_bbox[2] = max(x_max, fixed_bbox[2])
                fixed_bbox[3] = max(y_max, fixed_bbox[3])
        mask_reader.reset()

        fixed_bbox[0] = max(0, fixed_bbox[0] - crop_margin)
        fixed_bbox[1] = max(0, fixed_bbox[1] - crop_margin)
        fixed_bbox[2] = min(mask.shape[1], fixed_bbox[2] + crop_margin)
        fixed_bbox[3] = min(mask.shape[0], fixed_bbox[3] + crop_margin)
    elif keep_scale_and_center:
        bbox_size = [-1, -1]  # w, h
        for mask in mask_reader:
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox_size[0] = max(bbox_size[0], x_max - x_min)
            bbox_size[1] = max(bbox_size[1], y_max - y_min)
        mask_reader.reset()

    t = tqdm.tqdm(total=len(rgb_reader))
    for rgb, mask in zip(rgb_reader, mask_reader):
        t.update()
        savename = output / os.path.basename(rgb_reader.filename)
        if cache and savename.exists():
            continue

        # expanded_mask = expand_mask(mask, 0)
        expanded_mask = mask

        if keep_scale_and_fixed:
            x1, y1, x2, y2 = fixed_bbox[:]
            cropped_img = rgb[y1:y2, x1:x2]
            cropped_mask = expanded_mask[y1:y2, x1:x2]
        elif keep_scale_and_center:
            ys, xs = np.where(expanded_mask)
            cx = xs.mean()
            cy = ys.mean()
            x1 = int(max(0, cx - bbox_size[0] // 2))
            y1 = int(max(0, cy - bbox_size[1] // 2))
            x2 = int(min(rgb.shape[1] - 1, cx + bbox_size[0] // 2))
            y2 = int(min(rgb.shape[1] - 1, cy + bbox_size[1] // 2))
            cropped_img = rgb[y1:y2, x1:x2]
            cropped_mask = expanded_mask[y1:y2, x1:x2]
        else:
            cropped_img, cropped_mask = crop_to_mask(
                rgb, expanded_mask, margin=crop_margin
            )

        # 6. resize到1024x1024（保持比例，padding）
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
        padded_img = Image.new(
            "RGBA", (target_size, target_size), (0, 0, 0, 0)
        )
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
        if cropped_mask_np.max() == 1:
            alpha = (cropped_mask_np > 0.5).astype(np.uint8) * 255
        else:
            alpha = (cropped_mask_np > 127).astype(np.uint8) * 255
        rgba = np.array(padded_img)
        rgba[..., 3] = alpha
        out_img = Image.fromarray(rgba)
        out_img.save(savename)

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
