#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rotate imgdir and roidb
"""

from __future__ import print_function
import argparse
import os

import cPickle as pickle
import cv2
import numpy as np
from utils import rotate_boxes, rotate_image, rotate_kps


def main(args):
    if args.output is None:
        args.output = "output_%d" % args.angle
    output_imgdir = os.path.join(args.output, "images")
    if not os.path.exists(output_imgdir):
        os.makedirs(output_imgdir)

    print("loading %s" % args.roidb)
    with open(args.roidb, "r") as fid:
        roidb = pickle.load(fid)

    new_roidb = []
    for ind, roi in enumerate(roidb):
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(roidb)))
        imgfile = os.path.join(args.imgdir, os.path.basename(roi["image"]))
        img = cv2.imread(imgfile)
        img_r = rotate_image(img, args.angle)
        save_imgfile = os.path.join(
            output_imgdir, os.path.basename(roi["image"])
        )
        cv2.imwrite(save_imgfile, img_r)

        src_img_shape = img.shape
        dst_img_shape = img_r.shape
        roi["height"] = dst_img_shape[0]
        roi["width"] = dst_img_shape[1]

        boxes_r = rotate_boxes(
            roi["boxes"], args.angle, src_img_shape, dst_img_shape
        )
        boxes_r[:, 0] = np.maximum(0, boxes_r[:, 0])
        boxes_r[:, 1] = np.maximum(0, boxes_r[:, 1])
        boxes_r[:, 2] = np.minimum(roi["width"] - 1, boxes_r[:, 2])
        boxes_r[:, 3] = np.minimum(roi["height"] - 1, boxes_r[:, 3])

        boxes_r2 = boxes_r.copy()
        boxes_r2[:, 0] = np.minimum(boxes_r[:, 0], boxes_r[:, 2])
        boxes_r2[:, 1] = np.minimum(boxes_r[:, 1], boxes_r[:, 3])
        boxes_r2[:, 2] = np.maximum(boxes_r[:, 0], boxes_r[:, 2])
        boxes_r2[:, 3] = np.maximum(boxes_r[:, 1], boxes_r[:, 3])
        boxes_r = boxes_r2

        roi["boxes"] = boxes_r
        kps_r = rotate_kps(
            roi["keypoints"], args.angle, src_img_shape, dst_img_shape
        )
        roi["keypoints"] = kps_r
        new_roidb.append(roi)

    print("saved in %s" % args.output)
    savepkl = os.path.join(args.output, os.path.basename(args.roidb))
    with open(savepkl, "w") as fid:
        pickle.dump(new_roidb, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roidb", default=None, type=str, help="", required=True
    )
    parser.add_argument(
        "--imgdir", default=None, type=str, help="", required=True
    )
    parser.add_argument("--angle", default=90, type=int, help="rotate angle")
    parser.add_argument("--output", default=None, type=str, help="output")
    args = parser.parse_args()
    main(args)
