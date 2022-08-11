#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Optional
import typer
from pathlib import Path
import glob
try:
    import moviepy
    from moviepy.video.io import ImageSequenceClip
    from moviepy.video.fx.resize import resize
except ImportError as e:
    msg = "imgdir2video.py requires moviepy, please install moviepy"
    typer.echo(typer.style(msg, fg=typer.colors.RED))
    import sys; sys.exit()


def info(msg):
    typer.echo(typer.style(msg, fg=typer.colors.GREEN))


def warn(msg):
    typer.echo(typer.style(msg, fg=typer.colors.YELLOW))


def error(msg):
    typer.echo(typer.style(msg, fg=typer.colors.RED))


def main(
    imgdir: str,
    ext: Optional[str] = typer.Option('.jpg', help='image extension'),
    fps: Optional[int] = typer.Option(15, help='fps'),
    start: Optional[int] = typer.Option(0, help='start frame'),
    end: Optional[int] = typer.Option(-1, help='end frame'),
    factor: Optional[float] = typer.Option(1.0, help='video output size factor'),
    to_gif: Optional[bool] = typer.Option(False, help='generate gif'),
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Convert imgdir to video, including mp4 and gif."""
    imgdir = Path(imgdir)

    if output is None:
        suffix = ".gif" if to_gif else ".mp4"
        output = imgdir.name + f'_render{suffix}'
    imgfiles = sorted(glob.glob(str(imgdir / f"*{ext}")))
    imgfiles = imgfiles[start : end]

    info(f">> loading {imgdir} [{len(imgfiles)}]")
    clip = ImageSequenceClip.ImageSequenceClip(imgfiles, fps=fps)

    if factor != 1.0:
        info(f">> resize video {factor}")
        clip = clip.fx(resize, factor)

    info(f">> saving to {output}")
    if to_gif:
        clip.write_gif(output, fps=fps)
    else:
        clip.write_videofile(output)
    info(f">> done")


if __name__ == "__main__":
    typer.run(main)
