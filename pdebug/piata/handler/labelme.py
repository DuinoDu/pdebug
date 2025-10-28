import json
import os
from glob import glob
from typing import Dict, List

import numpy as np

from ..registry import ROIDB_REGISTRY
from ..type_cast import keypoints_to_points


def shapes_to_keypoints(shapes: Dict) -> np.ndarray:
    """Convert shapes to keypoints."""
    keypoints = []
    for shape in shapes:
        points = np.asarray(shape["points"], dtype=np.float32)
        num_kps = points.shape[0]
        assert points.shape[1] == 2
        vis = np.ones((num_kps, 1), dtype=np.float32)
        kps = np.concatenate((points, vis), axis=1).flatten()
        keypoints.append(kps)
    keypoints = np.asarray(keypoints)
    return keypoints


def keypoints_to_shapes(
    keypoints: np.ndarray,
    *,
    label: str = "0",
    group_id: str = None,
    shape_type: str = "polygon",
    flags: Dict = None,
) -> Dict:
    """Convert keypoints to shapes."""
    shapes = []
    all_points = keypoints_to_points(keypoints)
    for points in all_points:
        points = points.reshape(-1, 2).tolist()
        shape = {
            "label": label,
            "group_id": group_id,
            "shape_type": shape_type,
        }
        shape["points"] = points
        shape["flags"] = flags if flags else {}
        shapes.append(shape)
    return shapes


@ROIDB_REGISTRY.register(name="labelme")
def labelme_to_roidb(
    labelme: str,
    *,
    shapes2keypoints: bool = False,
    use_image_as_boxes: bool = False,
) -> List[Dict]:
    """Convert labelme result to roidb."""

    annofiles = sorted(glob(labelme + "/*.json"))
    roidb = []
    for annofile in annofiles:
        roi = dict()
        anno = json.load(open(annofile, "r"))
        assert "imagePath" in anno
        roi["image_name"] = os.path.basename(anno["imagePath"])
        if "imageHeight" in anno:
            roi["image_height"] = anno["imageHeight"]
        if "imageWidth" in anno:
            roi["image_width"] = anno["imageWidth"]

        if shapes2keypoints:
            assert "shapes" in anno, "`shapes` should be in anno jsonfile."
            roi["keypoints"] = shapes_to_keypoints(anno["shapes"])

        if use_image_as_boxes:
            boxes = [[0, 0, roi["image_width"] - 1, roi["image_height"] - 1]]
            roi["boxes"] = np.asarray(boxes, dtype=np.float32)
            roi["gt_classes"] = np.ones(len(boxes), dtype=np.float32)

        roidb.append(roi)
    return roidb


def save_to_labelme(
    roidb: List[Dict],
    outdir: str,
    *,
    version: str = "5.0.1",
    relative_imgdir: str = "images",
) -> None:
    """Save roidb to label json files."""
    os.makedirs(outdir, exist_ok=True)

    for roi in roidb:
        image_name = roi["image_name"]
        item = {"version": version, "flags": {}}

        if "keypoints" in roi:
            shapes = keypoints_to_shapes(roi["keypoints"])
            item["shapes"] = shapes

        if "image_height" in roi:
            item["imageHeight"] = roi["image_height"]
        if "image_width" in roi:
            item["imageWidth"] = roi["image_width"]
        item["imageData"] = None
        item["imagePath"] = f"{relative_imgdir}/{image_name}"

        savename = os.path.splitext(image_name)[0] + ".json"
        savefile = os.path.join(outdir, savename)
        with open(savefile, "w") as fid:
            json.dump(item, fid, indent=2)
