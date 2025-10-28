"""
SpatialTrackerV2 tracking infer node for OnePoseviaGen pipeline.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from pdebug.data_types import Camera
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.env import TORCH_INSTALLED
from pdebug.utils.fileio import do_system
from pdebug.visp import draw, rerun

import cv2
import numpy as np
import tqdm
import typer

if TORCH_INSTALLED:
    import torch
    import torchvision.transforms as T


@otn_manager.NODE.register(name="spatracker")
def main(
    input_path: str,
    repo: str,
    output: str = "tracking_results",
    vis_output: str = None,
    mask_path: str = None,
    fps: int = 1,
    max_frames: int = None,
    model_input_size: int = 30,
    grid_size: int = 50,
    topk: int = -1,
    cache: bool = True,
    vis_rerun: bool = False,
    rr_ip: str = None,
):
    """
    SpatialTrackerV2 tracking for OnePoseviaGen pipeline.

    Args:
        input_path: Path to RGB image folder or video path.
        mask_path: Path to mask folder
        repo: Path to OnePoseviaGen repository
        output: Output directory for tracking results
        fps: Frames per second for processing
        max_frames: Maximum frames to process
        grid_size: Grid size for tracking points
    """
    # Expand paths
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()
    if vis_output:
        vis_output = Path(vis_output).expanduser().resolve()
    if vis_rerun:
        cache = False
        assert rr_ip, "please provide rerun service ip address by --rr-ip."
        rerun.init_rerun(input_path.stem, rr_ip)
        rerun.rr_rgbd_blueprint()

    if not repo.exists():
        raise RuntimeError(f"SpaTrackerV2 repository not found at {repo}")
    sys.path.insert(0, str(repo))
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import (
        preprocess_image,
    )

    if output.exists() and (not cache):
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Get RGB and mask files
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

    if mask_path:
        mask_path = Path(mask_path).expanduser().resolve()
        mask_files = Input(str(mask_path), name="imgdir").get_reader().imgfiles
        mask_files.sort(key=lambda x: int(Path(x).stem))
        if len(mask_files) == 0:
            raise RuntimeError("No mask files found")
        mask_files = mask_files[::fps]
        if max_frames:
            mask_files = mask_files[:max_frames]

    # Load and process images
    frames = []
    frames_indexes = []
    for idx, img in enumerate(reader):
        if idx % fps > 0:
            continue
        frames.append(img)
        frames_indexes.append(idx)
        if max_frames and len(frames) > max_frames:
            break
    typer.echo(
        typer.style(f"Processing {len(frames)} frames", fg=typer.colors.GREEN)
    )

    if (
        cache
        and (output / "depth").exists()
        and len(os.listdir(output / "depth")) == len(frames)
    ):
        print(f"{output} exists, skip")
        return

    dummy_frames_len = model_input_size - len(frames) % model_input_size
    if dummy_frames_len:
        dummy_frames = [
            np.zeros_like(frames[-1]) for _ in range(dummy_frames_len)
        ]
        frames.extend(dummy_frames)

    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()

    # Resize to 336
    h, w = video_tensor.shape[2:]
    scale = 336 / min(h, w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
    else:
        new_h, new_w = h, w

    scale2 = 518 / max(h, w)
    target_width = int(new_w * scale2)
    video_tensor = preprocess_image(
        video_tensor, mode="crop", target_size=target_width, keep_ratio=True
    )
    B, C, H, W = video_tensor.shape
    all_video_tensor = video_tensor.reshape(-1, model_input_size, C, H, W)

    # Load VGGT4Track model
    vggt_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt_model.eval()
    tracker_model = Predictor.from_pretrained(
        "Yuxihenry/SpatialTrackerV2-Offline"
    )
    tracker_model.eval()

    depth_dir = output / "depth"
    depth_dir.mkdir(exist_ok=True)
    intrinsic_dir = output / "intrinsic"
    intrinsic_dir.mkdir(exist_ok=True)
    previous_depth_map_item = None

    t = tqdm.tqdm(total=all_video_tensor.shape[0], desc="processing")
    for ind, video_tensor in enumerate(all_video_tensor):
        t.update()
        if (
            cache
            and len(os.listdir(depth_dir)) >= (ind + 1) * model_input_size
        ):
            continue

        if ind == all_video_tensor.shape[0] - 1 and dummy_frames_len:
            video_tensor = video_tensor[: model_input_size - dummy_frames_len]
        if ind > 0:
            video_tensor = torch.cat(
                (all_video_tensor[ind - 1][-1:], video_tensor), dim=0
            )

        # Run VGGT4Track for depth and camera estimation
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = vggt_model.to("cuda")(
                    video_tensor[None].cuda() / 255
                )
                extrinsic, intrinsic = (
                    predictions["poses_pred"],
                    predictions["intrs"],
                )
                depth_map, depth_conf = (
                    predictions["points_map"][..., 2],
                    predictions["unc_metric"],
                )

        # release cuda
        torch.cuda.empty_cache()

        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()

        # Load mask and setup tracking points
        if mask_path:
            if ind > 0:
                mask_img = cv2.imread(
                    str(mask_files[ind * model_input_size - 1]), -1
                )
            else:
                mask_img = cv2.imread(
                    str(mask_files[ind * model_input_size]), -1
                )
            mask = cv2.resize(
                mask_img, (video_tensor.shape[3], video_tensor.shape[2])
            )

            # Get grid points
            from app_3rd.spatrack_utils.infer_track import get_points_on_a_grid

            frame_H, frame_W = video_tensor.shape[2:]
            grid_pts = get_points_on_a_grid(
                grid_size, (frame_H, frame_W), device="cuda"
            )

            # Filter points by mask
            grid_pts_int = grid_pts[0].long()
            mask_values = mask[
                grid_pts_int.cpu()[..., 1], grid_pts_int.cpu()[..., 0]
            ]
            grid_pts = grid_pts[:, mask_values]

            query_xyt = (
                torch.cat(
                    [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
                )[0]
                .cpu()
                .numpy()
            )

            # Run tracking
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    (
                        c2w_traj,
                        intrs,
                        point_map,
                        conf_depth,
                        track3d_pred,
                        track2d_pred,
                        vis_pred,
                        conf_pred,
                        video,
                    ) = tracker_model.cuda().forward(
                        video_tensor.cuda(),
                        depth=depth_tensor,
                        intrs=intrs,
                        extrs=extrs,
                        queries=query_xyt,
                        fps=1,
                        full_point=False,
                        iters_track=4,
                        query_no_BA=True,
                        fixed_cam=False,
                        stage=1,
                        unc_metric=depth_conf > 0.5,
                        support_frame=len(video_tensor) - 1,
                        replace_ratio=0.2,
                    )
        else:
            c2w_traj = None
            track3d_pred = None
            track2d_pred = None
            conf_pred = None
            point_map = None
            conf_depth = None
            vis_pred = None

        # recover to raw rgb size
        model_output_h, model_output_w = depth_map.shape[-2:]
        depth_map = T.Resize((h, w), T.InterpolationMode.NEAREST)(
            depth_map.unsqueeze(1)
        )
        scale_intrinsic_x = depth_map.shape[-2] / model_output_h
        scale_intrinsic_y = depth_map.shape[-1] / model_output_w
        intrinsic[:, :, 0, 0] *= scale_intrinsic_x
        intrinsic[:, :, 0, 2] *= scale_intrinsic_x
        intrinsic[:, :, 1, 1] *= scale_intrinsic_y
        intrinsic[:, :, 1, 2] *= scale_intrinsic_y

        if ind > 0:
            # Rescale depth_map to keep same scale with previous depth_map
            depth_map_scale = previous_depth_map_item / depth_map[0].squeeze()
            depth_map = depth_map[1:] * depth_map_scale
            intrs = intrs[1:]
            video_tensor = video_tensor[1:]
            intrinsic = intrinsic[:, 1:]
            if track3d_pred is not None:
                # TODO: depth_map is rescaled, but not for track3d_pred
                # Be cautious when using track3d_pred
                c2w_traj = c2w_traj[1:]
                track3d_pred = track3d_pred[1:]
                vis_pred = vis_pred[1:]
                conf_pred = conf_pred[1:]
        previous_depth_map_item = depth_map[-1].squeeze()

        if vis_rerun:
            camera = Camera(np.eye(4), intrs[0])
            for idx in range(len(depth_map)):
                rgb = frames[
                    ind * model_input_size : (ind + 1) * model_input_size
                ][idx]
                dep = depth_map[idx]
                if mask_path:
                    mask = cv2.imread(
                        mask_files[
                            ind
                            * model_input_size : (ind + 1)
                            * model_input_size
                        ][idx],
                        -1,
                    ).astype(bool)
                    dep[~mask] = 0
                rerun.rr_rgbd(
                    rgb, dep, camera, timestamp=idx + ind * model_input_size
                )

        if vis_output:
            rgb_fps = reader.fps if hasattr(reader, "fps") else 30
            depth_fps = rgb_fps // fps
            depth_writer = Output(
                vis_output / f"depth_{ind:03d}.mp4",
                name="video_ffmpeg",
                fps=depth_fps,
            ).get_writer()
            for rgb, sample in zip(
                frames[ind * model_input_size : (ind + 1) * model_input_size],
                depth_tensor,
            ):
                vis_depth = draw.depth(
                    sample, image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                depth_writer.write_frame(vis_depth)
            depth_writer.save()

        # Save results to npz
        if track3d_pred is not None:
            data_npz_load = {}
            data_npz_load["coords"] = (
                torch.einsum(
                    "tij,tnj->tni",
                    c2w_traj[:, :3, :3].cpu(),
                    track3d_pred[:, :, :3].cpu(),
                )
                + c2w_traj[:, :3, 3][:, None, :].cpu()
            ).numpy()
            data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
            data_npz_load["intrinsics"] = intrs.cpu().numpy()
            data_npz_load["depths"] = point_map[:, 2, ...].cpu().numpy()
            data_npz_load["video"] = (video_tensor).cpu().numpy() / 255
            data_npz_load["visibs"] = vis_pred.cpu().numpy()
            data_npz_load["confs"] = conf_pred.cpu().numpy()
            data_npz_load["confs_depth"] = conf_depth.cpu().numpy()
            npz_path = output / f"result_{ind}.npz"
            np.savez(str(npz_path), **data_npz_load)

        # Save depth images and intrinsic
        for frame_id, depth_map_save in zip(
            frames_indexes[
                ind * model_input_size : (ind + 1) * model_input_size
            ],
            depth_map.squeeze().cpu().numpy(),
        ):
            depth_map_mm = (depth_map_save * 1000).astype("uint16")
            depth_path = depth_dir / f"{frame_id:06d}.png"
            cv2.imwrite(str(depth_path), depth_map_mm)
            intrinsic_path = intrinsic_dir / f"{frame_id:06d}.json"
            intrinsic_data = intrinsic[0, frame_id % model_input_size].tolist()
            with open(intrinsic_path, "w") as f:
                json.dump(intrinsic_data, f, indent=2)

    typer.echo(
        typer.style(
            f"Tracking completed. Results saved to {output}",
            fg=typer.colors.GREEN,
        )
    )
    return str(output)


if __name__ == "__main__":
    typer.run(main)
