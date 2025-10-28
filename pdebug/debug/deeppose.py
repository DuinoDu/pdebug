import os
from typing import Optional, Tuple, Union

from pdebug.data_types import Tensor, x_to_ndarray
from pdebug.visp import draw

import cv2

__all__ = ["vis"]


def vis(
    pred: Tensor,
    target: Optional[Tensor] = None,
    image: Optional[Union[Tensor, str]] = None,
    output: Optional[str] = "./tmp_output",
    pred_color: Tuple[int] = (255, 0, 0),
    target_color: Tuple[int] = (0, 255, 0),
) -> None:
    """Visualize deeppose predict and target.

    Args:
        pred: predict keypoint coordinates, [B, N, 3]
        target: target keypoint coordinates, [B, N, 3]
        image: image, [B, H, W, 3], or image path list.

    Example:
        >>> from pdebug import deeppose
        >>> pred_kps = keypoint_head.decode(...)
        >>> target_kps = keypoint_head.decode(...)
        >>> deeppose.vis(pred_kps, target_kps, images)
    """
    pred = x_to_ndarray(pred)
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if image is not None:
        if isinstance(image[0], str):
            image = [cv2.imread(p) for p in image]
        image = x_to_ndarray(image)
    else:
        raise NotImplementedError
    assert pred.shape[0] == image.shape[0]

    if os.path.exists(output):
        os.system(f"rm -rf {output}")
    os.makedirs(output, exist_ok=True)

    if target is not None:
        target = x_to_ndarray(target)
        assert pred.shape[0] == target.shape[0]
        target = target.reshape(target.shape[0], -1)

    for ind, (pred_i, img) in enumerate(zip(pred, image)):
        img_vis = draw.skeleton_keyboard(img, pred_i, color=pred_color)
        if target is not None:
            target_i = target[ind]
            img_vis = draw.skeleton_keyboard(
                img_vis, target_i, color=target_color
            )

        savename = os.path.join(output, f"{ind:>06d}.jpg")
        cv2.imwrite(savename, img_vis)
