#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import os

import cPickle as pickle
import numpy as np

try:
    import commands
except Exception as e:
    import subprocess as commands


def fetch_image_wh(imgfile):
    cmd = "identify -format '%%w:%%h' %s" % imgfile
    (status, output) = commands.getstatusoutput(cmd)
    output = output.split("\n")[0]
    return int(output.split(":")[0]), int(output.split(":")[1])


def not_found(cocoGt, image_id):
    images_info = cocoGt["images"]
    file_name = [x["file_name"] for x in images_info if x["id"] == image_id]
    return len(file_name) == 0


def box_transform_inv(box):
    _box = [0, 0, 0, 0]
    _box[0] = box[0]
    _box[1] = box[1]
    _box[2] = box[0] + box[2] - 1
    _box[3] = box[1] + box[3] - 1
    return _box


def main(args):
    results = []
    results_dict = {}
    with open(args.cocoGt, "r") as fid:
        coco = json.load(fid)
        no_people = 0
        for ind, ann_img in enumerate(coco["images"]):
            if ind % 500 == 0:
                print("%d / %d" % (ind, len(coco["images"])))
            image_name = ann_img["file_name"]
            if image_name not in results_dict:
                results_dict[image_name] = {}
                (
                    results_dict[image_name]["width"],
                    results_dict[image_name]["height"],
                ) = fetch_image_wh(os.path.join(args.imgdir, image_name))
                results_dict[image_name]["boxes"] = []
                results_dict[image_name]["keypoints"] = []
                results_dict[image_name]["area"] = []
                results_dict[image_name]["iscrowd"] = []
            try:
                image_id = int(os.path.splitext(image_name)[0])
            except Exception as e:
                image_id = os.path.splitext(image_name)[0]

            anns = [
                x for x in coco["annotations"] if x["image_id"] == image_id
            ]
            if len(anns) == 0:
                no_people += 1
                if image_name in results_dict:
                    results_dict.pop(image_name)
                continue
            else:
                for ann in anns:
                    bbox = box_transform_inv(ann["bbox"])
                    bbox.extend([1.0])  # add bbox score
                    keypoints = ann["keypoints"]
                    if (
                        not args.keep_allzero
                        and np.count_nonzero(np.array(keypoints)) == 0
                    ):
                        continue
                    results_dict[image_name]["boxes"].append(bbox)
                    results_dict[image_name]["keypoints"].append(keypoints)
                    results_dict[image_name]["area"].append(ann["area"])
                    results_dict[image_name]["iscrowd"].append(ann["iscrowd"])
            if len(results_dict[image_name]["keypoints"]) == 0:
                results_dict.pop(image_name)

    results = [
        {
            "image": os.path.join(args.imgdir, x),
            "width": results_dict[x]["width"],
            "height": results_dict[x]["height"],
            "boxes": np.array(results_dict[x]["boxes"], dtype=np.float32),
            "keypoints": np.array(
                results_dict[x]["keypoints"], dtype=np.float32
            ),
            "gt_classes": np.ones(
                (len(results_dict[x]["boxes"]),), dtype=np.int32
            ),
            "flipped": False,
            "rotation": False,
            "upper_body": False,
            "degree": 0,
            "area": np.array(results_dict[x]["area"], dtype=np.float32),
            "iscrowd": np.array(results_dict[x]["iscrowd"]),
        }
        for x in results_dict.keys()
    ]

    print("no people: %d" % no_people)
    print(
        "have people: %d"
        % len([x for x in results if x["boxes"].shape[0] != 0])
    )

    if args.output is None:
        args.output = os.path.splitext(args.cocoGt)[0] + "_roidb.pkl"
    print("Writing results json to %s" % args.output)
    with open(args.output, "w") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cocoGt to roidb")
    parser.add_argument(
        "--cocoGt", default="", type=str, help="cocoGt", required=True
    )
    parser.add_argument(
        "--imgdir", default="", type=str, help="imgdir", required=True
    )
    parser.add_argument(
        "--output", default=None, type=str, help="output roidb pkl file"
    )
    parser.add_argument(
        "--keep_allzero",
        dest="keep_allzero",
        action="store_true",
        help="keep all zeros in keypoints, used when evaluate coco5k",
    )
    args = parser.parse_args()
    main(args)
