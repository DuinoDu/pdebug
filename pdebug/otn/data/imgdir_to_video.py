#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import math
import os
import shutil
from pathlib import Path
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input

import cv2
import tqdm
import typer

try:
    import moviepy
    from moviepy.editor import CompositeVideoClip, TextClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.fx.resize import resize
    from moviepy.video.io import ImageSequenceClip
except ImportError as e:
    moviepy = None


def info(msg):
    typer.echo(typer.style(msg, fg=typer.colors.GREEN))


def warn(msg):
    typer.echo(typer.style(msg, fg=typer.colors.YELLOW))


def error(msg):
    typer.echo(typer.style(msg, fg=typer.colors.RED))


@otn_manager.NODE.register(name="imgdir_to_video")
def imgdir_to_video(
    imgdir: str,
    ext: str = None,
    fps: int = 15,
    start: int = 0,
    end: int = -1,
    factor: float = 1.0,
    to_gif: bool = False,
    draw_filename_on_frame: bool = False,
    remove_imgdir: bool = False,
    output: str = None,
    cache: bool = False,
):
    """Convert imgdir to video, including mp4 and gif."""
    if moviepy is None:
        raise RuntimeError("moviepy is required.")

    imgdir = Path(imgdir)

    if output is None:
        suffix = ".gif" if to_gif else ".mp4"
        output = imgdir.name + f"_render{suffix}"

    if cache and os.path.exists(output):
        print(f"Found {output}, skip")
        return output

    if ext:
        typer.echo(typer.style("`ext` is deprecated", fg=typer.colors.YELLOW))

    # imgfiles = sorted(glob.glob(str(imgdir / f"*{ext}")))
    imgfiles = Input(str(imgdir), name="imgdir").get_reader().imgfiles
    imgfiles = imgfiles[start:end]

    info(f">> loading {imgdir} [{len(imgfiles)}]")
    clip = ImageSequenceClip.ImageSequenceClip(imgfiles, fps=fps)

    if factor != 1.0:
        info(f">> resize video {factor}")
        clip = clip.fx(resize, factor)

    if draw_filename_on_frame:
        print("generating title clips ...")
        title_clips = []
        for idx, imgfile in enumerate(imgfiles):
            title_clip = TextClip(
                filename=os.path.basename(imgfile), fontsize=15, color="red"
            )
            frame_duration = 1.0 / fps
            title_clip = title_clip.set_duration(frame_duration)
            title_clips.append(title_clip)
        title_clip = concatenate_videoclips(title_clips)
        clip = CompositeVideoClip([clip, title_clip])
        print("done.")

    info(f">> saving to {output}")
    if to_gif:
        clip.write_gif(output, fps=fps)
    else:
        clip.write_videofile(output)
    info(f">> done")

    if remove_imgdir:
        shutil.rmtree(imgdir)
    return output


@otn_manager.NODE.register(name="video_to_imgdir")
def video_to_imgdir(
    path: str,
    output: str = None,
    cache: bool = False,
    factor: float = 1.0,
):
    """Convert video to imgdir."""
    if moviepy is None:
        raise RuntimeError("moviepy is required.")

    if cache and os.path.exists(output):
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output
    os.makedirs(output, exist_ok=True)
    reader = Input(path, name="video").get_reader()
    t = tqdm.tqdm(total=len(reader), desc="video to images")
    for idx, frame in enumerate(reader):
        t.update()
        if factor != 1.0:
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * factor), int(frame.shape[0] * factor)),
            )
        savename = os.path.join(output, f"{idx:06d}.png")
        cv2.imwrite(savename, frame)
    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


if __name__ == "__main__":
    typer.run(imgdir_to_video)
