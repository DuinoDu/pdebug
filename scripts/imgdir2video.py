#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import math
import os
from pathlib import Path
from typing import Optional

import typer

try:
    import moviepy
    from moviepy.editor import CompositeVideoClip, TextClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.fx.resize import resize
    from moviepy.video.io import ImageSequenceClip
except ImportError as e:
    typer.echo(typer.style(e, fg=typer.colors.RED))
    msg = "imgdir2video.py requires moviepy, please install moviepy==1.0.3"
    typer.echo(typer.style(msg, fg=typer.colors.RED))
    import sys

    sys.exit()


def info(msg):
    typer.echo(typer.style(msg, fg=typer.colors.GREEN))


def warn(msg):
    typer.echo(typer.style(msg, fg=typer.colors.YELLOW))


def error(msg):
    typer.echo(typer.style(msg, fg=typer.colors.RED))


def main(
    imgdir: str,
    prefix: Optional[str] = typer.Option("", help="image name prefix"),
    ext: Optional[str] = typer.Option(".jpg", help="image extension"),
    fps: Optional[int] = typer.Option(15, help="fps"),
    start: Optional[int] = typer.Option(0, help="start frame"),
    end: Optional[int] = typer.Option(-1, help="end frame"),
    factor: Optional[float] = typer.Option(
        1.0, help="video output size factor"
    ),
    to_gif: Optional[bool] = typer.Option(False, help="generate gif"),
    draw_filename_on_frame: Optional[bool] = typer.Option(
        False, help="draw filename on frame"
    ),
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Convert imgdir to video, including mp4 and gif."""
    imgdir = Path(imgdir)

    if output is None:
        suffix = ".gif" if to_gif else ".mp4"
        output = imgdir.name + f"_render{suffix}"
    imgfiles = sorted(glob.glob(str(imgdir / f"{prefix}*{ext}")))
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
                filename=os.path.basename(imgfile),
                temptxt="I_dont_exist",
                fontsize=15,
                color="red",
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


if __name__ == "__main__":
    typer.run(main)
