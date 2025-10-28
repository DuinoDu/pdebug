import json
import logging
import os
from typing import Dict, List

import numpy as np
import tqdm

from ..registry import ROIDB_REGISTRY


@ROIDB_REGISTRY.register(name="llava_json")
def llava_json_to_roidb(
    jsonfile: str,
    image_root: str = None,
    strict: bool = True,
    quiet: bool = True,
    topk: int = -1,
) -> List[Dict]:
    """
    Convert llava json to roidb.

    Args:
        jsonfile: input json file.
        image_root: image root path.
    """
    with open(jsonfile, "r") as fid:
        data = json.load(fid)
    roidb = []

    if topk and topk > 0:
        data = data[:topk]

    if not quiet:
        t = tqdm.tqdm(total=len(data), desc="loading json")
    for anno in data:
        if not quiet:
            t.update()
        roi = dict()

        if "image" in anno:
            roi["image_name"] = anno["image"]
        elif "image_file" in anno:
            roi["image_name"] = anno["image_file"]
        else:
            if not quiet:
                print(f"[llava_json] image not found in {jsonfile}")
            if strict:
                continue
            roi["image_name"] = None
        if image_root and not os.path.exists(roi["image_name"]):
            roi["image_name"] = os.path.join(image_root, roi["image_name"])

        if not os.path.exists(roi["image_name"]) and strict:
            if not quiet:
                print(f"[llava_json] {roi['image_name']} not exists.")
            continue

        if "id" in anno:
            roi["id"] = anno["id"]

        if "conversations" in anno:
            # raw llava training data format
            roi["conversations"] = anno["conversations"]
        elif "prompt" in anno and "answer" in anno:
            roi["conversations"] = [
                {"from": "human", "value": anno["prompt"]},
                {"from": "gpt", "value": anno["answer"]},
            ]

        roidb.append(roi)
    return roidb
