import json
import logging
import os
from typing import List

import numpy as np

from ..registry import ROIDB_REGISTRY


@ROIDB_REGISTRY.register(name="vott")
def vott_to_roidb(vott: str) -> List:
    vott = os.path.abspath(vott)
    root = os.path.dirname(vott)
    annofiles = sorted(
        [
            os.path.join(root, x)
            for x in sorted(os.listdir(root))
            if x.endswith(".json")
        ]
    )

    roidb = []
    for annofile in annofiles:
        anno = json.load(open(annofile, "r"))
        assert "asset" in anno
        assert "regions" in anno

        roi = dict()
        roi["image_name"] = os.path.basename(anno["asset"]["path"])
        roi["image_height"] = anno["asset"]["size"]["height"]
        roi["image_width"] = anno["asset"]["size"]["width"]
        if "timestamp" in anno["asset"]:
            roi["image_timestamp"] = anno["asset"]["timestamp"]
        boxes = []
        for region in anno["regions"]:
            if "type" in region and region["type"] != "RECTANGLE":
                continue
            assert "person" in region["tags"]
            assert "boundingBox" in region
            x1 = region["boundingBox"]["left"]
            y1 = region["boundingBox"]["top"]
            w = region["boundingBox"]["width"]
            h = region["boundingBox"]["height"]
            boxes.append(
                [x1, y1, w + x1 - 1, y1 + h - 1]
            )  # roidb2coco will +1
        roi["boxes"] = np.asarray(boxes, dtype=np.float32)
        roi["gt_classes"] = np.ones(len(boxes), dtype=np.float32)
        roidb.append(roi)
    return roidb
