import os

from pdebug.data_types import Tensor, x_to_ndarray
from pdebug.visp import draw
from pdebug.visp.colormap import Colormap

import cv2
import numpy as np

from .heatmap import unnormalize

__all__ = ["vis_rgbd_semseg"]

_COUNT = 0


colors = None


def vis_rgbd_semseg(
    pred=None,
    target=None,
    image=None,
    depth=None,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
    image_isrgb=True,
    num_classes=40,
    output="auto",
    tmpdir="tmp_vis_rgbd_semseg",
    loss=None,
) -> None:
    """Visualize rgbd semseg predict and target."""
    global _COUNT
    global colors

    if output:
        os.makedirs(tmpdir, exist_ok=True)

    assert (pred is not None) or (target is not None)

    if pred is not None:
        pred = x_to_ndarray(pred)
        if pred.ndim == 2:
            pred = pred[None, :, :, :]
        batch_size = pred.shape[0]

    if target is not None:
        target = x_to_ndarray(target)
        if target.ndim == 2:
            target = target[None, :, :, :]
        batch_size = target.shape[0]

    if image is not None:
        image = x_to_ndarray(image)
        image = unnormalize(
            image,
            mean=image_mean,
            std=image_std,
            isrgb=image_isrgb,
            normal_method="pytorch",
        )
        if image.ndim == 3:
            image = image[None, :, :, :]

    if depth is not None:
        depth = x_to_ndarray(depth)
        if depth.ndim == 3:
            image = image[None, :, :, :]

    if loss is not None:
        loss = x_to_ndarray(loss)
        if loss.ndim == 0:
            loss = [float(loss) for _ in range(batch_size)]

    if not colors:
        colors = Colormap(num_classes + 1)

    all_vis = []
    for batch_idx in range(batch_size):
        vis_list = []

        if image is not None:
            image_i = image[batch_idx]
            vis_list.append(image_i)
        else:
            image_i = None

        if depth is not None:
            depth_i = depth[batch_idx]
            vis_depth_i = draw.depth(depth_i)
            vis_list.append(vis_depth_i)

        if pred is not None:
            pred_i = pred[batch_idx]
            vis_pred_i = draw.semseg(
                pred_i,
                image=image_i,
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
            vis_target_i = draw.semseg(
                target_i,
                image=image_i,
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

        if len(vis_list) > 1:
            vis_result = np.concatenate(vis_list, axis=1)
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

        if output == "auto":
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
