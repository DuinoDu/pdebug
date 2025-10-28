import os

from pdebug.data_types import Tensor, x_to_ndarray

import cv2
import numpy as np

__all__ = ["dump_image"]


def dump_image(x, savename="~/debug.png"):
    x = x_to_ndarray(x)
    if x.ndim == 3 and x.shape[0] < x.shape[2]:
        # convert CHW to HWC
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 4:
        raise NotImplementedError

    x = x.astype(np.uint8)

    savename = os.path.expanduser(savename)
    cv2.imwrite(savename, x)
    os.system(f"pwdscp {savename}")
