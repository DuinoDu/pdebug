import os
import sys
from pathlib import Path
from typing import Optional

from pdebug.data_types import PointcloudTensor
from pdebug.otn import manager as otn_manager
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


@otn_manager.NODE.register(name="depth_anything")
def depth_anything(
    image_path: str = "/mnt/c/Users/Admin/Pictures/dangdang1.jpg",
    output: str = "vis_depth.png",
    do_vis: bool = True,
):
    """Infer depth-anything."""
    global client
    assert Client is not None
    if client is None:
        client = Client(API_endpoint)

    result = client.predict(image_path, api_name="/on_submit")

    rgb_file = result[0][0]
    depth_vis_file = result[0][1]
    depth_raw_file = result[1]

    if do_vis:
        rgb = cv2.imread(rgb_file)
        depth_data = cv2.imread(depth_raw_file, cv2.IMREAD_UNCHANGED)

        vis_depth = draw.depth(
            depth_data, image=rgb, hist=True, vertical=False
        )
        cv2.imwrite(output, vis_depth)
        typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return depth_raw_file


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


@otn_manager.NODE.register(name="depth-anything-video")
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
    video_depth_anything.load_state_dict(
        torch.load(
            f"{repo}/checkpoints/video_depth_anything_{encoder}.pth",
            map_location="cpu",
        ),
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
            input_path, max_len, -1, max_res
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
            # TODO : how to save depth to png
            __import__("ipdb").set_trace()
            depth = (1 / depth * 1000).astype(np.uint16)  # convert to mm
            cv2.imwrite(savename, depth)

    if vis_output:
        video_name = os.path.basename(input_path)
        processed_video_path = os.path.join(
            vis_output, os.path.splitext(video_name)[0] + "_src.mp4"
        )
        depth_vis_path = os.path.join(
            vis_output, os.path.splitext(video_name)[0] + "_vis.mp4"
        )
        save_video(frames, processed_video_path, fps=fps)
        save_video(
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
