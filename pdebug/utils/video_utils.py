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

    try:
        from moviepy.editor import ImageClip, concatenate
    except ModuleNotFoundError:
        from moviepy import ImageClip, concatenate_videoclips

        concatenate = concatenate_videoclips

    imgfiles = sorted(
        [os.path.join(imgdir, x) for x in sorted(os.listdir(imgdir))]
    )
    if not output:
        output = os.path.basename(imgdir) + "_vis.mp4"

    clips = []
    for imgfile in imgfiles:
        clip = ImageClip(imgfile)
        if hasattr(clip, "set_duration"):
            clip = clip.set_duration(1)
        else:
            clip = clip.with_duration(1)
        clips.append(clip)

    video = concatenate(clips, method="compose")
    video.write_videofile(output, fps=fps)

    if remove_imgdir:
        shutil.rmtree(imgdir)
