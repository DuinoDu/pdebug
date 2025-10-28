import json
import logging
import os
from typing import List

from pdebug.utils.env import XMLTODICT_INSTALLED

import numpy as np
import tqdm

if XMLTODICT_INSTALLED:
    import xmltodict

from ..registry import ROIDB_REGISTRY


@ROIDB_REGISTRY.register(name="cvat_segmentation")
def cvat_images_to_roidb_segmentation(
    xmlfile: str, append_task_name=False
) -> List:
    """
    cvat format: CVAT for images 1.1

    Args:
        xmlfile: input xml file.
        extra_func: custom function to process xml dict.
    """
    assert XMLTODICT_INSTALLED, "Please install xmltodict."
    with open(xmlfile, "r", encoding="utf-8") as fid:
        data = xmltodict.parse(fid.read())
    assert data["annotations"]["version"] == "1.1", "Only test on 1.1 version."
    task_name = data["annotations"]["meta"]["task"]["name"]

    roidb = []
    for anno in data["annotations"]["image"]:
        roi = dict()
        roi["image_name"] = anno["@name"]
        roi["image_height"] = int(anno["@height"])
        roi["image_width"] = int(anno["@width"])

        if append_task_name:
            roi["task_name"] = task_name

        roi["segmentation"] = []
        roi["label"] = []

        if "polygon" not in anno:
            print(f"{anno['@name']} has not annotations, skip")
            continue

        if isinstance(anno["polygon"], dict):
            anno["polygon"] = [anno["polygon"]]
        for polygon in anno["polygon"]:
            try:
                poly_xy = [
                    [float(x) for x in p.split(",")]
                    for p in polygon["@points"].split(";")
                ]
            except Exception as e:
                __import__("ipdb").set_trace()
                raise e
            poly_xy = np.asarray(poly_xy).flatten().astype(np.int32).tolist()
            roi["segmentation"].append(poly_xy)
            roi["label"].append(polygon["@label"])
        roidb.append(roi)
    return roidb


@ROIDB_REGISTRY.register(name="cvat_keypoints")
def cvat_images_to_roidb_keypoints(xmlfile: str, quiet=False) -> List:
    """
    cvat format: CVAT 1.1

    Args:
        xmlfile: input xml file.
    """
    assert XMLTODICT_INSTALLED, "Please install xmltodict."
    with open(xmlfile, "r", encoding="utf-8") as fid:
        data = xmltodict.parse(fid.read())
    assert data["annotations"]["version"] == "1.1", "Only test on 1.1 version."

    def parse_points_dict(points_dict):
        # ['@label', '@source', '@occluded', '@points', '@z_order']
        points_xy = [x.split(",") for x in points_dict["@points"].split(";")]
        for i in range(len(points_xy)):
            points_xy[i] = [float(x) for x in points_xy[i]]
        points_xy = np.asarray(points_xy)
        num_points = points_xy.shape[0]
        points_vis = np.zeros((num_points, 1))
        points_vis.fill(2)
        keypoints = np.concatenate((points_xy, points_vis), axis=1).flatten()
        label = points_dict["@label"]
        return keypoints, label

    roidb = []
    if not quiet:
        t = tqdm.tqdm(
            total=len(data["annotations"]["image"]), desc="cvat to roidb"
        )
    for anno in data["annotations"]["image"]:
        if not quiet:
            t.update()
        roi = dict()
        roi["image_name"] = anno["@name"]
        roi["image_height"] = int(anno["@height"])
        roi["image_width"] = int(anno["@width"])
        if "points" not in anno:
            continue

        points_data = anno["points"]
        if isinstance(points_data, dict):
            roi["keypoints"], roi["label"] = parse_points_dict(points_data)
        elif isinstance(points_data, list):
            keypoints = []
            label = []
            for item in points_data:
                kps_item, label_item = parse_points_dict(item)
                keypoints.append(kps_item)
                label.append(label_item)
            roi["keypoints"] = np.asarray(keypoints).flatten()
            roi["label"] = label[0]
        # TODO: add extra info
        roidb.append(roi)
    return roidb
