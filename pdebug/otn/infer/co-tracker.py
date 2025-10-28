import copy
import os
import pickle
import shutil
import sys
from pathlib import Path

from pdebug.geometry import Vector2d
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.env import TORCH_INSTALLED
from pdebug.utils.fileio import save_json

import numpy as np
import tqdm
import typer
from PIL import Image

if TORCH_INSTALLED:
    import torch

DEFAULT_COTRACKER3_CKPT_ONLINE = (
    Path("~/.cache/install-x/co-tracker/checkpoints/scaled_online.pth")
    .expanduser()
    .resolve()
)
DEFAULT_COTRACKER3_CKPT_OFFLINE = (
    Path("~/.cache/install-x/co-tracker/checkpoints/scaled_offline.pth")
    .expanduser()
    .resolve()
)
UNITTEST_DATA = (
    Path("~/.cache/install-x/co-tracker/assets/apple.mp4")
    .expanduser()
    .resolve()
)


@otn_manager.NODE.register(name="cotracker")
def cotracker_main(
    input_path: str,
    grid_size: int = 10,
    grid_query_frame: int = 0,  # "Compute dense and grid tracks starting from this frame"
    output: str = "tmp_cotracker",
    vis_output: str = "tmp_cotracker_vis",
    cache: bool = False,
    backward_tracking: bool = True,  # only used in offline
    mask_path: str = None,  # mask path for the first frame, only used in offline
    model_name: str = "cotracker3_offline",
    unittest: bool = False,
    max_frames: int = None,
    fps: int = 1,
    model_input_size: int = 60,
):
    """Run co_tracker on video."""
    input_path = Path(input_path)
    output = Path(output)
    if vis_output:
        vis_output = Path(vis_output)

    if cache and output.exists():
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output
    output.mkdir(parents=True, exist_ok=True)

    assert input_path.is_file(), "Video file does not exist"

    assert model_name in [
        # "cotracker2_online", "cotracker2_offline",
        "cotracker3_online",
        "cotracker3_offline",
    ]

    try:
        from cotracker.predictor import (
            CoTrackerOnlinePredictor,
            CoTrackerPredictor,
        )
        from cotracker.utils.visualizer import Visualizer
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Install cotracker by bash $$INSTALL/co-tracker.sh"
        )

    if "online" in model_name:
        if not DEFAULT_COTRACKER3_CKPT_ONLINE.exists():
            model = torch.hub.load("facebookresearch/co-tracker", model_name)
        else:
            model = CoTrackerOnlinePredictor(
                checkpoint=DEFAULT_COTRACKER3_CKPT_ONLINE
            )
    else:
        if not DEFAULT_COTRACKER3_CKPT_OFFLINE.exists():
            model = torch.hub.load("facebookresearch/co-tracker", model_name)
        else:
            model = CoTrackerPredictor(
                checkpoint=DEFAULT_COTRACKER3_CKPT_OFFLINE
            )

    device = torch.device("cuda")
    model = model.to(device)
    all_pred_tracks, all_pred_visibility = [], []

    if unittest:
        import imageio.v3 as iio

        frames = iio.imread(UNITTEST_DATA, plugin="FFMPEG")  # plugin="pyav"
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W

        if "online" in model_name:
            model(video_chunk=video, is_first_step=True, grid_size=grid_size)
            for ind in range(0, video.shape[1] - model.step, model.step):
                pred_tracks, pred_visibility = model(
                    video_chunk=video[:, ind : ind + model.step * 2]
                )
                all_pred_tracks.append(pred_tracks)
                all_pred_visibility.append(pred_visibility)
            vis = Visualizer(save_dir=vis_output, pad_value=120, linewidth=3)
            vis.visualize(
                video,
                torch.cat(all_pred_tracks, dim=1),
                torch.cat(all_pred_visibility, dim=1),
                filename="online",
            )
        else:
            # video shape is [1, 50, 3, 720, 1296]
            pred_tracks, pred_visibility = model(
                video, grid_size=grid_size
            )  # B T N 2,  B T N 1
            vis = Visualizer(save_dir=vis_output, pad_value=120, linewidth=3)
            vis.visualize(
                video, pred_tracks, pred_visibility, filename="offline"
            )

        print("unittest passed")
        return

    window_frames = []
    reader = Input(input_path, name="video", to_rgb=True).get_reader()
    valid_frame_ids = []
    for ind, frame in enumerate(reader):
        if ind % fps != 0:
            continue
        if max_frames and len(window_frames) >= max_frames:
            break
        window_frames.append(frame)
        valid_frame_ids.append(ind)

    if "online" in model_name:
        # Iterating over video frames, processing one window at a time:

        dummy_frames_len = len(window_frames) % model.step
        if dummy_frames_len:
            dummy_frames = [
                np.zeros_like(window_frames[-1])
                for _ in range(dummy_frames_len)
            ]
            window_frames.extend(dummy_frames)

        num_windows = len(window_frames) // model.step
        t = tqdm.tqdm(total=num_windows, desc="online processing")
        for ind in range(0, len(window_frames) - model.step, model.step):
            t.update()
            # (1, T, 3, H, W)
            video_chunk = (
                torch.tensor(
                    window_frames[ind : ind + model.step * 2], device=device
                )
                .float()
                .permute(0, 3, 1, 2)[None]
            )
            if ind == 0:
                # init first
                model(
                    video_chunk=video_chunk,
                    is_first_step=True,
                    grid_size=grid_size,
                )
            pred_tracks, pred_visibility = model(video_chunk=video_chunk)
            all_pred_tracks.append(pred_tracks)
            all_pred_visibility.append(all_pred_visibility)

        raise NotImplementedError("Not supported online")
        if dummy_frames_len:
            all_pred_tracks = all_pred_tracks[-dummy_frames_len:]
            all_pred_visibility = all_pred_visibility[-dummy_frames_len:]
            window_frames = window_frames[-dummy_frames_len:]
        video = (
            torch.tensor(np.stack(window_frames), device=device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )
    else:
        if mask_path:
            mask_path = Path(mask_path).expanduser().resolve()
            mask_files = (
                Input(str(mask_path), name="imgdir").get_reader().imgfiles
            )
            mask_files.sort(key=lambda x: int(Path(x).stem))
            if len(mask_files) == 0:
                raise RuntimeError("No mask files found")
            mask_files = mask_files[::fps]
            if max_frames:
                mask_files = mask_files[:max_frames]

        dummy_frames_len = (
            model_input_size - len(window_frames) % model_input_size
        )
        if dummy_frames_len:
            dummy_frames = [
                np.zeros_like(window_frames[-1])
                for _ in range(dummy_frames_len)
            ]
            window_frames.extend(dummy_frames)

        num_chunk = len(window_frames) // model_input_size
        t = tqdm.tqdm(total=num_chunk, desc="offline processing")
        for ind in range(0, len(window_frames), model_input_size):
            t.update()
            # (1, T, 3, H, W)
            video_chunk = (
                torch.tensor(
                    window_frames[ind : ind + model_input_size], device=device
                )
                .float()
                .permute(0, 3, 1, 2)[None]
            )

            if mask_path:
                segm_mask = np.array(Image.open(os.path.join(mask_files[ind])))
                segm_mask = torch.from_numpy(segm_mask)[None, None]
            else:
                segm_mask = None

            pred_tracks, pred_visibility = model(
                video_chunk,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                segm_mask=segm_mask,
            )
            all_pred_tracks.append(pred_tracks)
            all_pred_visibility.append(pred_visibility)

        if dummy_frames_len:
            all_pred_tracks[-1] = all_pred_tracks[-1][:, :-dummy_frames_len]
            all_pred_visibility[-1] = all_pred_visibility[-1][
                :, :-dummy_frames_len
            ]
            window_frames = window_frames[:-dummy_frames_len]

    print("Tracks are computed")

    data = []
    for ind, (pred_tracks, pred_visibility) in enumerate(
        zip(all_pred_tracks, all_pred_visibility)
    ):
        kps = (
            torch.cat((pred_tracks, pred_visibility[:, :, :, None]), dim=3)
            .cpu()
            .numpy()
        )
        # kps.shape: [1, num_frames, num_kps, 3]
        data.append(
            {
                f"{ind:06d}": frame_kps.tolist()
                for ind, frame_kps in zip(valid_frame_ids, kps[0])
            }
        )

        if vis_output:
            vis = Visualizer(save_dir=vis_output, pad_value=120, linewidth=3)
            video = (
                torch.tensor(
                    np.stack(
                        window_frames[
                            ind
                            * model_input_size : (ind + 1)
                            * model_input_size
                        ]
                    ),
                    device=device,
                )
                .float()
                .permute(0, 3, 1, 2)[None]
            )
            vis.visualize(
                video,
                pred_tracks,
                pred_visibility,
                query_frame=grid_query_frame,
                filename=Path(input_path).stem + f"_{len(data)}",
            )

    save_json(data, output / "tracked_points.json")

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


def get_center_points(kps, topk):
    cx = (kps[:, 0].min() + kps[:, 0].max()) / 2
    cy = (kps[:, 1].min() + kps[:, 1].max()) / 2
    center = Vector2d(cx, cy)
    distance = [Vector2d.from_numpy(k[:2]).distance_to(center) for k in kps]
    idx = np.argsort(distance)[:topk]
    return kps[idx]


@otn_manager.NODE.register(name="video_kps_to_all")
def video_kps_to_all(
    path: str,
    anno_kps: str = None,
    output: str = "tmp_keypoints.pkl",
    cache: bool = False,
    vis_output: str = None,
    topk: int = None,
):
    """Run co_tracker on video, based on cvat annotate keypoints."""
    if cache and os.path.exists(output):
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output

    if vis_output:
        if os.path.exists(vis_output):
            shutil.rmtree(vis_output)
        os.makedirs(vis_output, exist_ok=True)

    if os.path.exists(os.path.join(path, "Frames.m4v")):
        path = os.path.join(path, "Frames.m4v")
    if not os.path.isfile(path):
        raise ValueError(f"Video file ({path}) does not exist")

    roidb = Input(anno_kps, name="cvat_keypoints").get_roidb()
    if topk and topk > 0:
        roidb = roidb[:topk]

    device = torch.device("cuda")
    tracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    tracker_model = tracker_model.to(device)

    window_frames = []
    reader = Input(path, name="video", to_rgb=True).get_reader()

    for frame in reader:
        window_frames.append(frame)
    video = (
        torch.tensor(np.stack(window_frames)).float().permute(0, 3, 1, 2)[None]
    )

    roidb_dict = {roi["image_name"]: roi for roi in roidb}
    t = tqdm.tqdm(total=len(roidb))
    for idx, roi in enumerate(roidb):
        t.update()
        image_name = roi["image_name"]
        frame_idx = int(os.path.basename(image_name)[:-4])
        kps = roi["keypoints"].reshape(-1, 3)
        kps = get_center_points(kps, topk=4)
        roidb_dict[image_name]["keypoints"] = kps.flatten()

        if idx > 0:
            prev_frame_idx = int(
                os.path.basename(roidb[idx - 1]["image_name"])[:-4]
            )
        else:
            prev_frame_idx = frame_idx

        if idx < len(roidb) - 1:
            next_frame_idx = int(
                os.path.basename(roidb[idx + 1]["image_name"])[:-4]
            )
        else:
            next_frame_idx = frame_idx

        for start, end in (
            [prev_frame_idx, frame_idx],
            [frame_idx, next_frame_idx],
        ):
            if start == end:
                continue
            cur_video = video[:, start:end].to(device)
            if end == frame_idx:
                cur_video = cur_video.flip(dims=(1,))

            query = torch.tensor(
                kps[:, :2], device=device, dtype=torch.float32
            )
            query_idx_tensor = torch.zeros(query.shape[0], device=device)
            query = torch.concat((query_idx_tensor[:, None], query), axis=1)

            pred_tracks, pred_visibility = tracker_model(
                cur_video,
                queries=query[None],
                backward_tracking=True,
            )
            if end == frame_idx:
                pred_tracks = pred_tracks.flip(dims=(1,))
                pred_visibility = pred_visibility.flip(dims=(1,))
                cur_video = cur_video.flip(dims=(1,))

            pred_xy = pred_tracks.cpu().numpy()[0]  # T, K, 2
            pred_vis = pred_visibility.cpu().numpy()[0]  # T, K

            for _id, (xy, vis) in enumerate(zip(pred_xy, pred_vis)):
                first_frame_idx = (
                    0 if start == frame_idx else pred_xy.shape[0] - 1
                )
                if _id == first_frame_idx:
                    continue
                image_name_pred = f"{start + _id:06d}.png"
                xy = xy[vis]
                vis = vis[vis].astype(xy.dtype)
                vis.fill(2)
                kps_pred = np.concatenate((xy, vis[:, None]), axis=1).flatten()

                if image_name_pred in roidb_dict:
                    roi_pred = roidb_dict[image_name_pred]
                    roi_pred["keypoints"] = np.concatenate(
                        (roi_pred["keypoints"], kps_pred), axis=0
                    )
                else:
                    roi_pred = copy.deepcopy(roi)
                    roi_pred["image_name"] = image_name_pred
                    roi_pred["keypoints"] = kps_pred
                    roidb_dict[image_name_pred] = roi_pred

            if vis_output:
                try:
                    from cotracker.utils.visualizer import Visualizer
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "Install cotracker by bash $$INSTALL/co-tracker.sh"
                    )

                vis = Visualizer(
                    save_dir=vis_output, pad_value=120, linewidth=3
                )
                savename = f"{start}_{end}"
                if end == frame_idx:
                    savename += "_reverse"
                vis.visualize(
                    cur_video,
                    pred_tracks,
                    pred_visibility,
                    query_frame=0,
                    filename=savename,
                )

    roidb = list(roidb_dict.values())
    roidb = sorted(roidb, key=lambda roi: roi["image_name"])
    for roi in roidb:
        roi["keypoints"] = roi["keypoints"][None, :]  # [K, 3] -> [1, K, 3]
    with open(output, "wb") as fid:
        pickle.dump(roidb, fid)

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


if __name__ == "__main__":
    typer.run(cotracker_main)
