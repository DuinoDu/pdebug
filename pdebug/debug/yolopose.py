import os

from pdebug.data_types import Tensor, x_to_ndarray
from pdebug.visp import draw

import cv2
import numpy as np

from .heatmap import heatmap

__all__ = ["scoremap"]

_COUNT = 0


def scoremap(
    x,
    savename="scoremap.png",
    batch_idx=0,
    score_idx=18,
    do_sigmoid=True,
    image=None,
):
    """
    Args:
        x: output from Pose module, three scale.
    """
    num_layers = len(x)
    vis_h, vis_w = 240, 320

    if isinstance(image, str) and os.path.exists(image):
        image = cv2.imread(image)[None, :, :, :]

    vis_all = None
    for xi in x:
        batch_size, num_anchor, feat_h, feat_w, pred_len = xi.shape
        scale = vis_h / feat_h
        if do_sigmoid:
            xi = xi.sigmoid()
        xi = x_to_ndarray(xi)
        score = xi[batch_idx, :, :, :, score_idx]
        vis_score = heatmap(score, image=image)
        if vis_score.shape[0] < vis_h:
            vis_score = cv2.resize(
                vis_score,
                (
                    int(vis_score.shape[1] * scale),
                    int(vis_score.shape[0] * scale),
                ),
            )
        if vis_all is None:
            vis_all = vis_score
        else:
            vis_all = np.concatenate((vis_all, vis_score), axis=0)
    cv2.imwrite(savename, vis_all)
