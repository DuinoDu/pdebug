import os

from pdebug.utils.env import MOVIEPY_INSTALLED
from pdebug.utils.video_utils import imgdir2video
from pdebug.visp import Colormap

import cv2
import numpy as np
import pytest


@pytest.mark.skipif(not MOVIEPY_INSTALLED, reason="moviepy is not installed")
def test_imgdir2video(tmpdir):
    colors = Colormap(10)
    for i in range(10):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = colors[i]
        cv2.imwrite(f"{tmpdir}/{i}.jpg", img)
    output = os.path.join(tmpdir, "vis.mp4")
    imgdir2video(tmpdir, output=output, fps=5, remove_imgdir=True)
