#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import os

import numpy as np


def not_found(cocoGt, image_id):
    images_info = cocoGt["images"]
    file_name = [x["file_name"] for x in images_info if x["id"] == image_id]
    return len(file_name) == 0


def main(args):
    results = []
    with open(args.cocoGt, "r") as fid:
        coco = json.load(fid)
        for ann in coco["annotations"]:
            result = dict()
            image_id = int(ann["image_id"])
            keypoints = ann["keypoints"]
            if np.count_nonzero(np.array(keypoints)) == 0:
                continue
            if len(keypoints) > 51:
                score = keypoints[-1]
                keypoints = keypoints[:51]
            elif len(keypoints) == 51:
                score = 1.0
            else:
                print("Bad keypoints len: %d" % len(keypoints))
                return
            result = {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "score": score,
            }
            if "bbox" in ann:
                result["bbox"] = ann["bbox"]
            results.append(result)
    if args.output is None:
        args.output = os.path.splitext(args.cocoGt)[0] + "_cocoDt.json"
    print("saved to %s" % args.output)
    with open(args.output, "w") as f:
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cocoGt to cocoDt")
    parser.add_argument(
        "--cocoGt", default="", type=str, help="cocoGt", required=True
    )
    parser.add_argument(
        "--output", default=None, type=str, help="output cocoDt file"
    )
    args = parser.parse_args()
    main(args)
