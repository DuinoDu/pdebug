import os

from pdebug.data_types import Tensor, x_to_ndarray
from pdebug.visp import draw

import cv2
import numpy as np

from .heatmap import unnormalize

__all__ = ["vis_depth"]

_COUNT = 0


def vis_depth(
    pred=None,
    target=None,
    image=None,
    image_mean=(0, 0, 0),
    image_std=(1.0 / 255, 1.0 / 255, 1.0 / 255),
    image_isrgb=True,
    output=None,
    tmpdir="tmp_vis_depth",
    loss=None,
    min_depth=0.0,
) -> None:
    """Visualize depth predict and target."""
    global _COUNT
    if output:
        os.makedirs(tmpdir, exist_ok=True)

    assert (pred is not None) or (target is not None)

    if pred is not None:
        pred = x_to_ndarray(pred)
        if pred.ndim == 3:
            pred = pred[None, :, :, :]
        batch_size = pred.shape[0]

    if target is not None:
        target = x_to_ndarray(target)
        if target.ndim == 3:
            target = target[None, :, :, :]
        batch_size = target.shape[0]

    if image is not None:
        image = x_to_ndarray(image)
        image = unnormalize(
            image, mean=image_mean, std=image_std, isrgb=image_isrgb
        )
        if image.ndim == 3:
            image = image[None, :, :, :]

    if loss is not None:
        loss = x_to_ndarray(loss)
        if loss.ndim == 0:
            loss = [float(loss) for _ in range(batch_size)]

    all_vis = []
    for batch_idx in range(batch_size):
        if image is not None:
            image_i = image[batch_idx]
        else:
            image_i = None
        vis_list = []
        if pred is not None:
            pred_i = pred[batch_idx]
            vis_pred_i = draw.depth(
                pred_i,
                image=image_i,
                vertical=False,
                hist=True,
                min_depth=min_depth,
            )
            vis_pred_i = cv2.putText(
                vis_pred_i,
                "pred",
                (100, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            vis_list.append(vis_pred_i)
        if target is not None:
            target_i = target[batch_idx]
            vis_target_i = draw.depth(
                target_i,
                image=image_i,
                vertical=False,
                hist=True,
                min_depth=min_depth,
            )
            vis_target_i = cv2.putText(
                vis_target_i,
                "gt",
                (100, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            vis_list.append(vis_target_i)
        if len(vis_list) == 2:
            vis_result = np.concatenate(vis_list, axis=0)
        else:
            vis_result = vis_list[0]

        if loss is not None:
            vis_result = cv2.putText(
                vis_result,
                f"loss: {loss[batch_idx]: .4f}",
                (100, 150),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if output is "auto":
            savename = f"{_COUNT:06d}_{batch_idx}.jpg"
        else:
            savename = output

        if savename and tmpdir:
            savename = os.path.join(tmpdir, savename)
        if savename:
            cv2.imwrite(savename, vis_result)
        all_vis.append(vis_result)

    _COUNT += 1
    return all_vis
