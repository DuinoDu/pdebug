import os
import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pdebug.data_types import PointcloudTensor
from pdebug.piata import Input
from pdebug.utils.env import TORCH_INSTALLED
from pdebug.utils.fileio import do_system
from pdebug.visp import draw

import cv2
import numpy as np
import typer

if TORCH_INSTALLED:
    import torch

try:
    from gradio_client import Client
except Exception as e:
    Client = None


client = None
# get from https://liheyoung-depth-anything.hf.space/?view=api
API_endpoint = "https://liheyoung-depth-anything.hf.space/--replicas/l4hs0/"


@lru_cache(maxsize=1)
def _load_depth_anything_model():
    assert TORCH_INSTALLED
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor, device


def depth_anything(
    image_path: str = "/mnt/c/Users/Admin/Pictures/dangdang1.jpg",
    output: str = "vis_depth.png",
    do_vis: bool = True,
):
    """Infer depth-anything."""
    assert TORCH_INSTALLED
    from PIL import Image

    model, processor, device = _load_depth_anything_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_data = prediction.detach().cpu().numpy()
    depth_raw_file = output
    if not do_vis:
        np.save(output, depth_data)
        return str(output)

    if do_vis:
        rgb = cv2.imread(str(image_path))
        vis_depth = draw.depth(
            depth_data, image=rgb, hist=False, vertical=False
        )
        cv2.imwrite(output, vis_depth)
        typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return str(depth_raw_file)


def depth_to_pcd(
    depth, rgb, fx=470.4, fy=470.4, cx=None, cy=None, savename=None
):
    # CC: update depth_to_pcd to use cx and cy
    height, width = rgb.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    if not cx:
        cx = width / 2
    if not cy:
        cy = height / 2
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = np.array(depth)
    points = np.stack(
        (np.multiply(x, z), np.multiply(y, z), z), axis=-1
    ).reshape(-1, 3)
    colors = np.array(rgb).reshape(-1, 3) / 255.0
    pcd = PointcloudTensor(points, color=colors)
    if savename:
        pcd.to_ply(savename)
    return pcd


def _save_depth_anything_video_fallback(
    frames, output_video_path, fps=10, is_depths=False, grayscale=False
):
    """Write mp4 output without relying on Video-Depth-Anything imageio args."""
    output_video_path = str(output_video_path)
    data = np.asarray(frames)
    if is_depths:
        import matplotlib.cm as cm

        if data.ndim != 3:
            raise ValueError(
                f"expected depth frames with shape T,H,W; got {data.shape}"
            )
        d_min, d_max = float(np.min(data)), float(np.max(data))
        denom = d_max - d_min
        if denom <= 1e-8:
            depth_norm = np.zeros(data.shape, dtype=np.uint8)
        else:
            depth_norm = ((data - d_min) / denom * 255).astype(np.uint8)
        if grayscale:
            data = np.repeat(depth_norm[..., None], 3, axis=-1)
        else:
            colormap = np.array(cm.get_cmap("inferno").colors)
            data = (colormap[depth_norm] * 255).astype(np.uint8)
    if data.ndim != 4:
        raise ValueError(
            f"expected video frames with shape T,H,W,C; got {data.shape}"
        )
    height, width = data.shape[1:3]
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_video_path}")
    try:
        for frame in data:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _save_depth_anything_video(save_video, frames, output_path, **kwargs):
    try:
        save_video(frames, output_path, **kwargs)
    except TypeError as exc:
        if "macro_block_size" not in str(exc):
            raise
        _save_depth_anything_video_fallback(frames, output_path, **kwargs)


def depth_anything_video(
    input_path: str = None,
    output: str = "tmp_depth_anything_video",
    vis_output: str = "tmp_depth_anything_video_vis",
    repo: str = None,
    unittest: bool = False,
    device: str = "cuda",
    encoder: str = "vitl",  # vits
    save_npz: bool = False,
    save_exr: bool = False,
    save_png: bool = False,
    metric: bool = False,
):
    """Infer video-depth-anything."""
    output = Path(output)
    output.mkdir(exist_ok=True)
    if vis_output:
        vis_output = Path(vis_output)
        vis_output.mkdir(exist_ok=True)

    repo = Path(repo)
    sys.path.insert(0, str(repo))
    from utils.dc_utils import read_video_frames, save_video
    from video_depth_anything.video_depth import VideoDepthAnything

    assert TORCH_INSTALLED

    if unittest:
        input_path = repo / "./assets/example_videos/davis_rollercoaster.mp4"
    else:
        input_path = Path(input_path)

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    video_depth_anything = VideoDepthAnything(**model_configs[encoder])
    checkpoint_path = (
        f"{repo}/checkpoints/metric_video_depth_anything_{encoder}.pth"
        if metric
        else f"{repo}/checkpoints/video_depth_anything_{encoder}.pth"
    )
    video_depth_anything.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"),
        strict=True,
    )
    video_depth_anything = video_depth_anything.to(device).eval()

    input_size = 518
    max_res = 1280
    max_len = -1
    target_fps = 10
    fp32 = False

    if input_path.is_file():
        frames, target_fps = read_video_frames(
            str(input_path), max_len, -1, max_res
        )
    else:
        frames = [
            rgb
            for rgb in Input(
                input_path, name="imgdir", to_rgb=True
            ).get_reader()
        ]
        if max(frames[0].shape) > max_res:
            scale = max_res / max(frames[0].shape)
            height = round(frames[0].shape[0] * scale)
            width = round(frames[0].shape[1] * scale)
            height -= 1 if height % 2 != 0 else 0
            width -= 1 if width % 2 != 0 else 0
            for i in range(len(frames)):
                frames[i] = cv2.resize(frames[i], (width, height))
        frames = np.stack(frames, axis=0)

    depths, fps = video_depth_anything.infer_video_depth(
        frames, target_fps, input_size=input_size, device=device, fp32=fp32
    )

    if save_png:
        assert input_path.is_dir()
        reader = Input(input_path, name="imgdir", to_rgb=True).get_reader()
        assert len(reader) == len(depths)

        original_width, original_height = None, None
        for imgname, depth in zip(reader.imgfiles, depths):
            savename = output / os.path.basename(imgname)
            assert str(savename).endswith(".png")

            if original_width is None:
                rgb = reader.imread(imgname)
                original_height, original_width = rgb.shape[:2]
            depth = cv2.resize(
                depth,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if metric:
                depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max)
            else:
                finite = np.isfinite(depth) & (depth > 1e-8)
                if finite.any():
                    inv_depth = np.zeros_like(depth, dtype=np.float32)
                    inv_depth[finite] = 1.0 / depth[finite]
                    scale = 1000.0 / np.median(inv_depth[finite])
                    depth_mm = np.clip(
                        inv_depth * scale, 0, np.iinfo(np.uint16).max
                    )
                else:
                    depth_mm = np.zeros_like(depth, dtype=np.float32)
            cv2.imwrite(str(savename), depth_mm.astype(np.uint16))

    if vis_output:
        video_name = os.path.basename(input_path)
        processed_video_path = os.path.join(
            vis_output, os.path.splitext(video_name)[0] + "_src.mp4"
        )
        depth_vis_path = os.path.join(
            vis_output, os.path.splitext(video_name)[0] + "_vis.mp4"
        )
        _save_depth_anything_video(save_video, frames, processed_video_path, fps=fps)
        _save_depth_anything_video(
            save_video,
            depths, depth_vis_path, fps=fps, is_depths=True, grayscale=False
        )

    if save_npz:
        depth_npz_path = os.path.join(
            output, os.path.splitext(video_name)[0] + "_depths.npz"
        )
        np.savez_compressed(depth_npz_path, depths=depths)
    if save_exr:
        depth_exr_dir = os.path.join(
            output, os.path.splitext(video_name)[0] + "_depths.exr"
        )
        os.makedirs(depth_exr_dir, exist_ok=True)
        import Imath
        import OpenEXR

        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

    return output


if __name__ == "__main__":
    typer.run(depth_anything_video)
