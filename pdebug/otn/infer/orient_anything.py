import math
import os
import sys
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.env import TORCH_INSTALLED
from pdebug.utils.fileio import download_file
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer
from PIL import Image

if TORCH_INSTALLED:
    import torch

DINO_SMALL = "facebook/dinov2-small"
DINO_BASE = "facebook/dinov2-base"
DINO_LARGE = "facebook/dinov2-large"
DINO_GIANT = "facebook/dinov2-giant"


@otn_manager.NODE.register(name="orient_anything")
def main(
    input_path: str = None,
    output: str = "tmp_orient",
    vis_output: str = "tmp_orient_vis",
    repo: str = None,
    unittest: bool = False,
    cache: bool = True,
    topk: int = -1,
    do_rm_bkg: bool = False,
    do_infer_aug: bool = False,
    device: str = "cuda:0",
):
    """Inference of Orient Anything."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    if vis_output:
        vis_output = Path(vis_output)
        vis_output.mkdir(exist_ok=True)
        writer = Output(
            vis_output / "visualization.mp4", name="video_ffmpeg"
        ).get_writer()
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))

    from huggingface_hub import hf_hub_download
    from inference import get_3angle, get_3angle_infer_aug
    from transformers import AutoImageProcessor
    from utils import (
        background_preprocess,
        overlay_images_with_scaling,
        render_3D_axis,
    )
    from vision_tower import DINOv2_MLP

    if unittest:
        import imageio.v3 as iio

        url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        os.makedirs("/tmp/unittest", exist_ok=True)
        download_file(url, "/tmp/unittest/lenna.png")
        input_path = "/tmp/unittest"

    input_path = Path(input_path)
    if input_path.is_dir():
        reader = Input(
            input_path, name="imgdir", topk=topk, to_rgb=True
        ).get_reader()
    else:
        reader = Input(
            input_path, name="video", topk=topk, to_rgb=True
        ).get_reader()

    ckpt_path = hf_hub_download(
        repo_id="Viglong/Orient-Anything",
        filename="croplargeEX2/dino_weight.pt",
        repo_type="model",
        resume_download=True,
    )
    dino = DINOv2_MLP(
        dino_mode="large",
        in_dim=1024,
        out_dim=360 + 180 + 180 + 2,
        evaluate=True,
        mask_dino=False,
        frozen_back=False,
    )
    dino.eval()
    print("model create")
    dino.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dino = dino.to(device)
    print("weight loaded")
    val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE)

    def infer_func(img, do_rm_bkg, do_infer_aug):
        """
        infer_func migrated from repo.

        Args:
            img: rgb np.ndarray
            do_rm_bkg: boo, remove background
            do_infer_aug: test time augmentation

        Returns:
            img: visualize result, 512x512
            azimuth: 0 ~ 360
            polar: -90 ~ 90
            rotation: -90 ~ 90
            confidence: 0 ~ 1

        """
        origin_img = Image.fromarray(img)
        if do_infer_aug:
            rm_bkg_img = background_preprocess(origin_img, True)
            angles = get_3angle_infer_aug(
                origin_img, rm_bkg_img, dino, val_preprocess, device
            )
        else:
            rm_bkg_img = background_preprocess(origin_img, do_rm_bkg)
            angles = get_3angle(rm_bkg_img, dino, val_preprocess, device)

        phi = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = angles[2]
        confidence = float(angles[3])
        if confidence > 0.5:
            render_axis = render_3D_axis(phi, theta, gamma)
            res_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
        else:
            res_img = img
        return [
            res_img,
            round(float(angles[0]), 2),
            round(float(angles[1]), 2),
            round(float(angles[2]), 2),
            round(float(angles[3]), 2),
        ]

    typer.echo(
        typer.style(f"loading {len(reader)} images", fg=typer.colors.GREEN)
    )
    t = tqdm.tqdm(total=len(reader))
    for rgb in reader:
        t.update()
        vis_result, *angles, conf = infer_func(rgb, do_rm_bkg, do_infer_aug)

        savename = output / os.path.basename(reader.filename)
        if cache and savename.exists():
            continue
        if vis_output:
            if unittest:
                savename = vis_output / os.path.basename(reader.filename)
                vis_result.save(savename)
            if hasattr(vis_result, "convert"):
                vis_res = np.asarray(vis_result.convert("RGB"))[
                    :, :, ::-1
                ].copy()
            else:
                vis_res = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
            writer.write_frame(vis_res)

    if vis_output:
        writer.save()

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
