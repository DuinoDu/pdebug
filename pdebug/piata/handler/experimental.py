"""Experimental handlers."""
import json
import os
from typing import Dict, List, Union

import cv2
import numpy as np

from ..registry import ROIDB_REGISTRY


@ROIDB_REGISTRY.register(name="mmpose")
def mmpose_to_roidb(pred_json: Union[str, List]) -> List[Dict]:
    """Convert mmpose pred result to roidb."""

    if isinstance(pred_json, str):
        with open(pred_json, "r") as fid:
            data = json.load(fid)
    else:
        assert isinstance(pred_json, list)
        data = pred_json

    roidb = []
    for batch in data:
        for image_path, pred in zip(batch["image_paths"], batch["preds"]):
            img = cv2.imread(image_path)
            roi = {
                "image_name": os.path.basename(image_path),
                "image_height": img.shape[0],
                "image_width": img.shape[1],
                "boxes": np.array(
                    [[0, 0, img.shape[1] - 1, img.shape[0] - 1]],
                    dtype=np.float32,
                ),
                "trackid": np.array([0], dtype=np.int),
            }
            kps = np.asarray(pred)
            kps[:, 2] = 2.0
            kps = kps.reshape((1, -1))
            roi["keypoints"] = kps
            roidb.append(roi)
    return roidb
