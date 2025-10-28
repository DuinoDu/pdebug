import os
import shutil

import numpy as np

__all__ = ["imgdir2video"]


def imgdir2video(
    imgdir: str,
    output: str = None,
    fps: int = 24,
    remove_imgdir: bool = False,
) -> str:
    try:
        import moviepy
    except ModuleNotFoundError as e:
        print("`imgdir2video` need `moviepy`, please install by pip.")
        raise e

    from moviepy.editor import ImageClip, concatenate

    imgfiles = sorted(
        [os.path.join(imgdir, x) for x in sorted(os.listdir(imgdir))]
    )
    if not output:
        output = os.path.basename(imgdir) + "_vis.mp4"

    clips = []
    for imgfile in imgfiles:
        clips.append(ImageClip(imgfile).set_duration(1))

    video = concatenate(clips, method="compose")
    video.write_videofile(output, fps=fps)

    if remove_imgdir:
        shutil.rmtree(imgdir)
