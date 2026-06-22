#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rotate imgdir
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
        args.output = args.imgdir + "_%d" % args.angle
    output_imgdir = os.path.join(args.output, "images")
    if not os.path.exists(output_imgdir):
        os.makedirs(output_imgdir)

    imgfiles = sorted(
        [
            os.path.join(args.imgdir, x)
            for x in sorted(os.listdir(args.imgdir))
            if x.endswith(args.ext)
        ]
    )
    for ind, imgfile in enumerate(imgfiles):
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(imgfiles)))
        img = cv2.imread(imgfile)
        img_r = rotate_image(img, args.angle)
        save_imgfile = os.path.join(output_imgdir, os.path.basename(imgfile))
        cv2.imwrite(save_imgfile, img_r)

    print("saved in %s" % args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imgdir", default=None, type=str, help="", required=True
    )
    parser.add_argument("--ext", default="jpg", type=str, help="image ext")
    parser.add_argument("--angle", default=90, type=int, help="rotate angle")
    parser.add_argument("--output", default=None, type=str, help="output")
    args = parser.parse_args()
    main(args)
