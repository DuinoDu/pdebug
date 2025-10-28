#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import shutil
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

"""
rename method:
1. keepname: keep name
2. concat: concat updir name
3. index: rename using index
"""


def main(args):
    images = []
    args.imgdir = os.path.abspath(args.imgdir)
    for path, dirs, files in os.walk(args.imgdir, followlinks=True):
        images.extend(
            [os.path.join(path, x) for x in files if x.endswith(args.ext)]
        )
    assert args.rename_method in ("keepname", "concat", "index"), (
        "Unknown rename method: %s" % args.rename_method
    )
    images = sorted(images)

    if args.copy:
        create_img = lambda x, y: shutil.copyfile(x, y)
    else:
        create_img = lambda x, y: os.symlink(x, y)

    if args.output is None:
        args.output = "images"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.imglist is not None:
        imglist = [x.strip() for x in open(args.imglist, "r").readlines()]
    if args.roidb is not None:
        print("loading %s" % args.roidb)

        with open(args.roidb, "rb") as fid:
            try:
                roidb = pickle.load(fid)
            except Exception as e:
                roidb = pickle.load(fid, encoding="iso-8859-1")
            try:
                imglist = [
                    os.path.basename(roi["image_name"]) for roi in roidb
                ]
            except KeyError as e:
                imglist = [os.path.basename(roi["image"]) for roi in roidb]
    if args.max > -1:
        images = images[: args.max]
    if args.min > -1:
        images = images[args.min :]

    for ind, imagefile in enumerate(images):
        if args.imglist is not None or args.roidb is not None:
            if os.path.basename(imagefile) not in imglist:
                continue
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(images)))
        if args.rename_method == "keepname":
            dst_name = os.path.basename(imagefile)
        elif args.rename_method == "concat":
            index = -1 * (args.depth + 1)
            dst_name = "_".join(imagefile.split("/")[index:])
        elif args.rename_method == "index":
            dst_name = "%08d.%s" % (ind, args.ext)
        create_img(imagefile, os.path.join(args.output, dst_name))
    print("Saved in %s" % args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect images from dir, recursively"
    )
    parser.add_argument(
        "--imgdir",
        default=None,
        type=str,
        help="input image dir, recursively",
        required=True,
    )
    parser.add_argument(
        "--rename_method",
        default="keepname",
        type=str,
        help="image rename method, keepname | concat | index",
    )
    parser.add_argument("--output", default=None, type=str, help="output dir")
    parser.add_argument("--ext", default="jpg", type=str, help="image ext")
    parser.add_argument(
        "--copy",
        dest="copy",
        action="store_true",
        help="if copy image, default is link",
    )
    parser.add_argument(
        "--depth",
        default=1,
        type=int,
        help="concat dir depth, only used when rename_method is concat",
    )
    parser.add_argument(
        "--imglist", default=None, type=str, help="choose imglist"
    )
    parser.add_argument(
        "--roidb", default=None, type=str, help="choose imglist from roidb"
    )
    parser.add_argument(
        "--min", default=-1, type=int, help="start image index"
    )
    parser.add_argument("--max", default=-1, type=int, help="end image index")
    args = parser.parse_args()
    main(args)
